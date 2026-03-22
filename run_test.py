#!/usr/bin/env python3
"""Run a single GenAI OTel conformance test against Weaver live-check.

This is the main entry point for running conformance tests. It orchestrates:

  1. Starting a mock LLM server (if not already running) that simulates
     provider APIs (OpenAI, Anthropic, etc.) on localhost.
  2. Starting a Weaver live-check instance that receives OTel telemetry
     via gRPC and validates it against the semantic conventions registry.
  3. Running the test program, which uses an instrumented LLM client to
     call the mock server and emit telemetry to Weaver.
  4. Stopping Weaver, collecting results, and updating the per-test data
     file used by the conformance dashboard.

Usage:
    python run_test.py <test-name> [weaver-args...]

Test name format: {lang}-{lib}-{ecosystem}
    e.g. python-openai-otelcontrib, js-openai-openllmetry,
         java-openai-otelcontrib, dotnet-extensions-ai-native

    lang:      python | js | java | dotnet
    lib:       the library under test (may contain hyphens, e.g. spring-ai)
    ecosystem: the instrumentation source — otelcontrib, openllmetry,
               openinference, or native

Requires:
    - Python 3.12+ (for mock server)
    - Language-specific toolchain for the test being run
    - Network access on first run if the pinned Weaver release is not already
      installed locally or available on PATH
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from genai_otel_conformance.result_helpers import (
    build_signal_statuses,
    build_span_type_statuses,
    present_attributes,
)
from genai_otel_conformance.results import (
    GENAI_EVENT_TYPES,
    GENAI_METRIC_TYPES,
    LANGUAGE_DISPLAY_NAMES,
    SPAN_TYPE_ORDER,
    SPAN_TYPE_SPECS,
    parse_result_dir,
    split_test_name,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR / "tests"
VERSIONS_FILE = SCRIPT_DIR / "versions.env"
MOCK_SERVER_PORT = 8080


def _load_version_pins() -> dict[str, str]:
    """Load shared external version pins from the repository root."""
    try:
        content = VERSIONS_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read version pins file: {VERSIONS_FILE}") from exc

    pins: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise RuntimeError(f"Invalid version pin line in {VERSIONS_FILE}: {raw_line!r}")
        pins[key.strip()] = value.strip().strip('"').strip("'")

    return pins


VERSION_PINS = _load_version_pins()
WEAVER_VERSION = VERSION_PINS["WEAVER_VERSION"]
SEMCONV_VERSION = VERSION_PINS["SEMCONV_VERSION"]


def _normalize_version(version: str) -> str:
    """Normalize optional leading-v version strings for comparisons."""
    return version[1:] if version.startswith("v") else version


def _weaver_binary_name() -> str:
    """Return the platform-specific Weaver binary name."""
    return "weaver.exe" if sys.platform == "win32" else "weaver"


def _weaver_asset_name() -> str:
    """Return the pinned Weaver release asset name for the current platform."""
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-pc-windows-msvc.zip"
    if sys.platform == "linux" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-unknown-linux-gnu.tar.xz"
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "weaver-aarch64-apple-darwin.tar.xz"
    if sys.platform == "darwin" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-apple-darwin.tar.xz"
    raise RuntimeError(
        f"Unsupported platform for managed Weaver install: {sys.platform} / {platform.machine()}"
    )


def _find_weaver_binary(search_root: Path) -> Path | None:
    """Return the first Weaver binary found under the given directory."""
    for path in search_root.rglob(_weaver_binary_name()):
        if path.is_file():
            return path
    return None


def _weaver_version(binary: str | Path) -> str:
    """Return the normalized version reported by a Weaver binary, or an empty string."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""

    output = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"\bv?(\d+\.\d+\.\d+)\b", output)
    return match.group(1) if match else ""


def _download_file(url: str, destination: Path) -> None:
    """Download a file to disk."""
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_weaver() -> Path:
    """Ensure the pinned Weaver version is available and return its binary path."""
    expected_version = _normalize_version(WEAVER_VERSION)

    system_weaver = shutil.which(_weaver_binary_name())
    if system_weaver and _weaver_version(system_weaver) == expected_version:
        return Path(system_weaver)

    cache_root = Path(os.environ.get(
        "WEAVER_CACHE",
        Path.home() / ".cache" / "otel-conformance" / "weaver",
    ))
    install_dir = cache_root / WEAVER_VERSION.replace("/", "_")
    cached_binary = _find_weaver_binary(install_dir) if install_dir.exists() else None
    if cached_binary and _weaver_version(cached_binary) == expected_version:
        return cached_binary

    asset_name = _weaver_asset_name()
    download_url = (
        f"https://github.com/open-telemetry/weaver/releases/download/{WEAVER_VERSION}/{asset_name}"
    )
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=str(install_dir.parent),
        prefix=f"weaver-{expected_version}-",
    ) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / asset_name
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        print(f"=== Downloading Weaver {WEAVER_VERSION} ===")
        _download_file(download_url, archive_path)

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_dir)
        else:
            with tarfile.open(archive_path, "r:*") as archive:
                archive.extractall(extract_dir)

        extracted_binary = _find_weaver_binary(extract_dir)
        if extracted_binary is None:
            raise RuntimeError(f"Downloaded Weaver archive did not contain {_weaver_binary_name()}")

        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.copytree(extract_dir, install_dir)

    cached_binary = _find_weaver_binary(install_dir)
    if cached_binary is None:
        raise RuntimeError(f"Installed Weaver binary not found under {install_dir}")
    if sys.platform != "win32":
        cached_binary.chmod(cached_binary.stat().st_mode | 0o111)
    installed_version = _weaver_version(cached_binary)
    if installed_version != expected_version:
        raise RuntimeError(
            f"Installed Weaver version mismatch: expected {WEAVER_VERSION}, found {installed_version or 'unknown'}"
        )
    return cached_binary

# ── Language / ecosystem configuration ──────────────────────────────

_LANG_DIRS = LANGUAGE_DISPLAY_NAMES


# ── Data structures ──────────────────────────────────────────────────


class TestCommandResult(NamedTuple):
    found: bool
    exit_code: int


@dataclass(frozen=True)
class LanguageAdapter:
    install_dependencies: Callable[[str, str], None]
    prebuild_test: Callable[[str], None]
    run_test: Callable[[str, str, dict[str, str]], TestCommandResult]
    list_tests: Callable[[], list[str]]


# ── Test data generation ────────────────────────────────────────────


def _data_path_from_test_name(test_name: str) -> Path:
    """Compute the data file path from a test name."""
    lang, lib, eco = split_test_name(test_name)
    return TESTS_DIR / lang / lib / f"data-{eco}.json"


def _results_dir_from_test_name(test_name: str) -> Path:
    """Compute the results directory path from a test name."""
    lang, lib, eco = split_test_name(test_name)
    return TESTS_DIR / lang / lib / "results" / eco


def _prepare_results_dir(result_dir: Path) -> None:
    """Ensure the result directory starts empty for a fresh Weaver run."""
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


def _has_weaver_output(result_dir: Path) -> bool:
    """Return whether Weaver wrote any JSON output files for this run."""
    return any(result_dir.glob("**/*.json"))


def generate_single_test_data(test_name: str) -> tuple[Path, dict] | None:
    """Generate data for a single test from its results directory.

    Returns (data_file_path, data_dict) or None if no relevant data.
    """
    result_dir = _results_dir_from_test_name(test_name)
    r = parse_result_dir(result_dir, test_name)
    if r is None or not r.has_data:
        return None

    all_present = present_attributes(r)
    has_genai = any(a.startswith("gen_ai.") for a in all_present)
    event_statuses = build_signal_statuses(GENAI_EVENT_TYPES, r.seen_events, r.detected_events)
    metric_statuses = build_signal_statuses(GENAI_METRIC_TYPES, r.seen_metrics, r.detected_metrics)
    has_genai_signals = any(status == "present" for status in event_statuses.values()) or any(
        status == "present" for status in metric_statuses.values()
    )
    span_types = build_span_type_statuses(r, SPAN_TYPE_ORDER, SPAN_TYPE_SPECS)

    if not span_types and not has_genai_signals:
        return None

    path = _data_path_from_test_name(test_name)

    data: dict[str, dict[str, str] | dict[str, dict[str, str]]] = {
        "events": event_statuses,
        "metrics": metric_statuses,
    }
    if span_types:
        data["span_types"] = span_types

    return path, data


# ── Test discovery and execution ────────────────────────────────────


def _gradle_cmd(test_dir: Path) -> list[str]:
    """Return the Gradle wrapper command for the given test directory."""
    if sys.platform == "win32":
        return [str((test_dir / "gradlew.bat").resolve())]
    return ["./gradlew"]


def _npm_cmd() -> str:
    """Return the platform-specific npm executable name."""
    return "npm.cmd" if sys.platform == "win32" else "npm"


def _uv_cmd() -> str:
    """Return the platform-specific uv executable name or exit with guidance."""
    uv = shutil.which("uv.exe" if sys.platform == "win32" else "uv")
    if uv:
        return uv

    print("ERROR: uv is required to install Python test dependencies.", file=sys.stderr)
    print("Install it and retry: https://docs.astral.sh/uv/getting-started/installation/", file=sys.stderr)
    sys.exit(1)


def _install_with_uv(*install_args: str, label: str) -> None:
    """Install Python dependencies into the current interpreter using uv."""
    print(f"=== Installing {label} ===")
    subprocess.run(
        [_uv_cmd(), "pip", "install", "--python", sys.executable, *install_args],
        cwd=SCRIPT_DIR,
        check=True,
    )


def _python_install_dependencies(lib: str, ecosystem: str) -> None:
    _install_with_uv("-e", "tests/python", label="shared Python test support")
    _install_with_uv(
        "-r",
        f"tests/python/{lib}/requirements-{ecosystem}.txt",
        label=f"Python test dependencies for {lib}/{ecosystem}",
    )


def _noop_install_dependencies(_lib: str, _ecosystem: str) -> None:
    return None


def _noop_prebuild(_lib: str) -> None:
    return None


def _js_prebuild_test(lib: str) -> None:
    test_dir = Path(f"tests/js/{lib}")
    npm = _npm_cmd()
    print(f"=== Installing JS dependencies in {test_dir} ===")
    subprocess.run([npm, "install", "--silent"], cwd=test_dir, check=True)


def _java_prebuild_test(lib: str) -> None:
    test_dir = Path(f"tests/java/{lib}")
    gradle = _gradle_cmd(test_dir)
    print(f"=== Pre-building Java project in {test_dir} ===")
    subprocess.run([*gradle, "classes"], cwd=test_dir, check=True)


def _dotnet_prebuild_test(lib: str) -> None:
    test_dir = Path(f"tests/dotnet/{lib}")
    print(f"=== Pre-building .NET project in {test_dir} ===")
    subprocess.run(["dotnet", "build"], cwd=test_dir, check=True)


def _python_run_test(lib: str, ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    test_file = Path(f"tests/python/{lib}/test_{ecosystem}.py")
    if not test_file.is_file():
        return TestCommandResult(False, 0)
    proc = subprocess.run([sys.executable, str(test_file)], env=env)
    return TestCommandResult(True, proc.returncode)


def _js_run_test(lib: str, ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    test_dir = Path(f"tests/js/{lib}")
    if not test_dir.is_dir():
        return TestCommandResult(False, 0)
    npm = _npm_cmd()
    test_proc = subprocess.run([npm, "run", f"test:{ecosystem}"], cwd=test_dir, env=env)
    return TestCommandResult(True, test_proc.returncode)


def _java_run_test(lib: str, _ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    test_dir = Path(f"tests/java/{lib}")
    if not test_dir.is_dir():
        return TestCommandResult(False, 0)
    gradle = _gradle_cmd(test_dir)
    proc = subprocess.run([*gradle, "run"], cwd=test_dir, env=env)
    return TestCommandResult(True, proc.returncode)


def _dotnet_run_test(lib: str, _ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    test_dir = Path(f"tests/dotnet/{lib}")
    if not test_dir.is_dir():
        return TestCommandResult(False, 0)
    proc = subprocess.run(["dotnet", "run"], cwd=test_dir, env=env)
    return TestCommandResult(True, proc.returncode)


def _python_list_tests() -> list[str]:
    tests: list[str] = []
    for test_file in sorted(Path("tests/python").glob("*/test_*.py")):
        lib = test_file.parent.name
        ecosystem = test_file.stem.removeprefix("test_")
        tests.append(f"python-{lib}-{ecosystem}")
    return tests


def _js_list_tests() -> list[str]:
    tests: list[str] = []
    for test_file in sorted(Path("tests/js").glob("*/test_*.ts")):
        lib = test_file.parent.name
        ecosystem = test_file.stem.removeprefix("test_")
        tests.append(f"js-{lib}-{ecosystem}")
    return tests


def _java_list_tests() -> list[str]:
    tests: list[str] = []
    for build_file in sorted(Path("tests/java").glob("*/build.gradle.kts")):
        lib = build_file.parent.name
        tests.append(f"java-{lib}-otelcontrib")
    return tests


def _dotnet_list_tests() -> list[str]:
    tests: list[str] = []
    for csproj in sorted(Path("tests/dotnet").glob("*/*.csproj")):
        lib = csproj.parent.name
        tests.append(f"dotnet-{lib}-native")
    return tests


LANGUAGE_ADAPTERS: dict[str, LanguageAdapter] = {
    "python": LanguageAdapter(
        install_dependencies=_python_install_dependencies,
        prebuild_test=_noop_prebuild,
        run_test=_python_run_test,
        list_tests=_python_list_tests,
    ),
    "js": LanguageAdapter(
        install_dependencies=_noop_install_dependencies,
        prebuild_test=_js_prebuild_test,
        run_test=_js_run_test,
        list_tests=_js_list_tests,
    ),
    "java": LanguageAdapter(
        install_dependencies=_noop_install_dependencies,
        prebuild_test=_java_prebuild_test,
        run_test=_java_run_test,
        list_tests=_java_list_tests,
    ),
    "dotnet": LanguageAdapter(
        install_dependencies=_noop_install_dependencies,
        prebuild_test=_dotnet_prebuild_test,
        run_test=_dotnet_run_test,
        list_tests=_dotnet_list_tests,
    ),
}


def run_test_cmd(name: str, env: dict[str, str]) -> TestCommandResult:
    """Run the test command.

    Returns whether the test was found and the test command's exit code.
    Weaver violations are handled separately from test-process failures.
    """
    lang, lib, eco = split_test_name(name)
    adapter = LANGUAGE_ADAPTERS.get(lang)
    if adapter is None:
        return TestCommandResult(False, 0)
    return adapter.run_test(lib, eco, env)


def list_available_tests() -> list[str]:
    """Discover all available test names."""
    tests: list[str] = []
    for lang in _LANG_DIRS:
        tests.extend(LANGUAGE_ADAPTERS[lang].list_tests())
    return tests


def _print_available_tests() -> None:
    """Print the current list of runnable tests to stderr."""
    print("Available tests:", file=sys.stderr)
    for test_name in list_available_tests():
        print(f"  {test_name}", file=sys.stderr)


# ── Health check helper ─────────────────────────────────────────────


def is_healthy(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def wait_for_health(url: str, timeout: int, label: str, proc: subprocess.Popen | None = None) -> None:
    for i in range(1, timeout + 1):
        if is_healthy(url):
            print(f"{label} ready after {i}s")
            return
        if proc and proc.poll() is not None:
            print(f"ERROR: {label} process died during startup", file=sys.stderr)
            sys.exit(1)
        time.sleep(1)
    print(f"ERROR: {label} failed to become ready after {timeout}s", file=sys.stderr)
    sys.exit(1)


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_test.py <test-name> [weaver-args...]", file=sys.stderr)
        sys.exit(1)

    test_name = sys.argv[1]
    extra_weaver_args = sys.argv[2:]

    print(f"Test: {test_name}")

    # ── Configuration from environment ──────────────────────────────

    try:
        lang, lib, eco = split_test_name(test_name)
    except ValueError:
        print(f"ERROR: Invalid test name '{test_name}'", file=sys.stderr)
        print("Expected format: <lang>-<library>-<ecosystem>", file=sys.stderr)
        _print_available_tests()
        sys.exit(1)

    LANGUAGE_ADAPTERS[lang].install_dependencies(lib, eco)

    weaver_port = random.randint(10000, 60000) & ~1  # even base
    admin_port = weaver_port + 1

    # ── Local registry cache ────────────────────────────────────────

    semconv_cache_root = Path(os.environ.get(
        "SEMCONV_CACHE",
        Path.home() / ".cache" / "otel-conformance" / "semconv",
    ))
    semconv_cache = semconv_cache_root / SEMCONV_VERSION.replace("/", "_")
    registry = os.environ.get("REGISTRY")
    if not registry:
        model_dir = semconv_cache / "model"
        if model_dir.is_dir():
            registry = str(model_dir)
        else:
            print(f"=== Caching semantic conventions registry ({SEMCONV_VERSION}) ===")
            semconv_cache.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--branch", SEMCONV_VERSION, "--depth", "1", "-q",
                 "https://github.com/open-telemetry/semantic-conventions.git",
                 str(semconv_cache)],
                check=True,
            )
            registry = str(model_dir)

    # ── Mock server ─────────────────────────────────────────────────

    mock_url = f"http://127.0.0.1:{MOCK_SERVER_PORT}"
    mock_proc: subprocess.Popen | None = None
    weaver_proc: subprocess.Popen | None = None

    try:
        if not is_healthy(f"http://127.0.0.1:{MOCK_SERVER_PORT}/health"):
            _install_with_uv("-e", "tests/mock-server", label="shared mock server dependencies")
            print(f"=== Starting mock server on port {MOCK_SERVER_PORT} ===")
            mock_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "tests" / "mock-server" / "mock_server" / "server.py"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(MOCK_SERVER_PORT),
                ],
            )
            wait_for_health(
                f"http://127.0.0.1:{MOCK_SERVER_PORT}/health",
                30,
                "Mock server",
                mock_proc,
            )
        else:
            print(f"Mock server already running on port {MOCK_SERVER_PORT}")

        # ── Pre-build (compile before starting weaver) ──────────────
        # Weaver uses an inactivity timeout; long builds (e.g. Gradle)
        # can cause it to shut down before the test sends any data.

        LANGUAGE_ADAPTERS[lang].prebuild_test(lib)

        # ── Start weaver ────────────────────────────────────────────

        test_results_dir = Path(f"tests/{lang}/{lib}/results/{eco}").resolve()
        _prepare_results_dir(test_results_dir)

        print(f"=== Starting weaver live-check for: {test_name} (ports {weaver_port}/{admin_port}) ===")
        weaver_bin = ensure_weaver()
        weaver_cmd = [str(weaver_bin), "registry", "live-check"]
        if registry:
            weaver_cmd += ["-r", registry]
        weaver_cmd += [
            "--format", "json",
            "--output", str(test_results_dir),
            "--otlp-grpc-port", str(weaver_port),
            "--admin-port", str(admin_port),
            "--inactivity-timeout", "30",
        ] + extra_weaver_args

        weaver_proc = subprocess.Popen(weaver_cmd)

        print("Waiting for weaver to be ready...")
        wait_for_health(f"http://localhost:{admin_port}/health", 60, "Weaver", weaver_proc)

        test_env = {
            **os.environ,
            "MOCK_LLM_URL": mock_url,
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{weaver_port}",
        }
        print(f"=== Running test: {test_name} ===")

        test_run = run_test_cmd(test_name, test_env)
        if not test_run.found:
            print(f"ERROR: Could not find test '{test_name}'", file=sys.stderr)
            _print_available_tests()
            sys.exit(1)

        # ── Stop weaver ─────────────────────────────────────────────

        time.sleep(1)
        if weaver_proc.poll() is None:
            try:
                urllib.request.urlopen(
                    urllib.request.Request(f"http://localhost:{admin_port}/stop", method="POST"),
                    timeout=5,
                )
            except Exception:
                weaver_proc.terminate()

        weaver_exit = weaver_proc.wait()
        print(f"=== Weaver exit code: {weaver_exit} ===")
        print(f"=== Results in: {test_results_dir} ===")

        if test_run.exit_code != 0:
            print(
                f"ERROR: Test command exited with code {test_run.exit_code}.",
                file=sys.stderr,
            )
            sys.exit(test_run.exit_code or 1)

        fresh_result = parse_result_dir(test_results_dir, test_name)
        has_weaver_output = _has_weaver_output(test_results_dir)
        has_weaver_stats = fresh_result is not None and fresh_result.statistics is not None

        if not has_weaver_output:
            print(
                f"ERROR: Weaver produced no JSON output for test: {test_name}",
                file=sys.stderr,
            )
            sys.exit(1)

        if weaver_exit != 0 and not has_weaver_stats:
            print(
                "ERROR: Weaver exited non-zero before writing statistics.",
                file=sys.stderr,
            )
            sys.exit(weaver_exit or 1)
        if weaver_exit != 0:
            print(
                "Note: Weaver returned a non-zero exit code because violations were reported; continuing with captured statistics.",
                file=sys.stderr,
            )

        # ── Update the per-test data file ───────────────────────────

        print("=== Updating test data file ===")
        result = generate_single_test_data(test_name)
        if result is not None:
            path, data = result
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"Updated {path}")
        else:
            print(f"ERROR: No relevant data for test: {test_name}", file=sys.stderr)
            sys.exit(1)

    finally:
        if weaver_proc and weaver_proc.poll() is None:
            print(f"Stopping weaver (PID {weaver_proc.pid})...")
            weaver_proc.terminate()
            weaver_proc.wait()
        if mock_proc and mock_proc.poll() is None:
            print(f"Stopping mock server (PID {mock_proc.pid})...")
            mock_proc.terminate()
            mock_proc.wait()


if __name__ == "__main__":
    main()
