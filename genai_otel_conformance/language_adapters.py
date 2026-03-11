"""Per-language test adapters: build, run, and discover conformance tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from genai_otel_conformance import REPO_ROOT, TESTS_DIR


class TestCommandResult(NamedTuple):
    found: bool
    exit_code: int


@dataclass(frozen=True)
class LanguageAdapter:
    install_dependencies: Callable[[str, str], None]
    prebuild_test: Callable[[str], None]
    run_test: Callable[[str, str, dict[str, str]], TestCommandResult]
    list_tests: Callable[[], list[str]]


# ── Shared helpers ──────────────────────────────────────────────────


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


def install_with_uv(*install_args: str, label: str) -> None:
    """Install Python dependencies into the current interpreter using uv."""
    print(f"=== Installing {label} ===")
    subprocess.run(
        [_uv_cmd(), "pip", "install", "--python", sys.executable, *install_args],
        cwd=REPO_ROOT,
        check=True,
    )


# ── Python adapter ──────────────────────────────────────────────────


def _python_install_dependencies(lib: str, ecosystem: str) -> None:
    install_with_uv("-e", "tests/python", label="shared Python test support")
    install_with_uv(
        "-r",
        f"tests/python/{lib}/requirements-{ecosystem}.txt",
        label=f"Python test dependencies for {lib}/{ecosystem}",
    )


def _python_run_test(lib: str, ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    test_file = TESTS_DIR / "python" / lib / f"test_{ecosystem}.py"
    if not test_file.is_file():
        return TestCommandResult(False, 0)
    proc = subprocess.run([sys.executable, str(test_file)], env=env)
    return TestCommandResult(True, proc.returncode)


def _python_list_tests() -> list[str]:
    return _list_tests_from_matches(
        "python",
        "*/test_*.py",
        lambda path: path.stem.removeprefix("test_"),
    )


# ── JS adapter ──────────────────────────────────────────────────────


def _js_workspace_dir() -> Path | None:
    workspace_dir = TESTS_DIR / "js"
    return workspace_dir if (workspace_dir / "package.json").is_file() else None


def _js_prebuild_test(lib: str) -> None:
    workspace_dir = _js_workspace_dir()
    test_dir = workspace_dir if workspace_dir is not None else TESTS_DIR / "js" / lib
    npm = _npm_cmd()
    print(f"=== Installing JS dependencies in {test_dir} ===")
    subprocess.run([npm, "install", "--silent"], cwd=test_dir, check=True)


def _js_run_test(lib: str, ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    workspace_dir = _js_workspace_dir()
    test_dir = TESTS_DIR / "js" / lib
    if not test_dir.is_dir():
        return TestCommandResult(False, 0)
    npm = _npm_cmd()
    if workspace_dir is not None:
        test_proc = subprocess.run(
            [npm, "--workspace", f"./{lib}", "run", f"test:{ecosystem}"],
            cwd=workspace_dir,
            env=env,
        )
    else:
        test_proc = subprocess.run([npm, "run", f"test:{ecosystem}"], cwd=test_dir, env=env)
    return TestCommandResult(True, test_proc.returncode)


def _js_list_tests() -> list[str]:
    return _list_tests_from_matches(
        "js",
        "*/test_*.ts",
        lambda path: path.stem.removeprefix("test_"),
    )


# ── Java adapter ────────────────────────────────────────────────────


def _java_prebuild_test(lib: str) -> None:
    test_dir = TESTS_DIR / "java" / lib
    gradle = _gradle_cmd(test_dir)
    print(f"=== Pre-building Java project in {test_dir} ===")
    subprocess.run([*gradle, "classes"], cwd=test_dir, check=True)


def _java_run_test(lib: str, _ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    test_dir = TESTS_DIR / "java" / lib
    if not test_dir.is_dir():
        return TestCommandResult(False, 0)
    gradle = _gradle_cmd(test_dir)
    proc = subprocess.run([*gradle, "run"], cwd=test_dir, env=env)
    return TestCommandResult(True, proc.returncode)


def _java_list_tests() -> list[str]:
    return _list_tests_from_matches("java", "*/build.gradle.kts", lambda _path: "otelcontrib")


# ── .NET adapter ────────────────────────────────────────────────────


def _dotnet_prebuild_test(lib: str) -> None:
    test_dir = TESTS_DIR / "dotnet" / lib
    print(f"=== Pre-building .NET project in {test_dir} ===")
    subprocess.run(["dotnet", "build"], cwd=test_dir, check=True)


def _dotnet_run_test(lib: str, _ecosystem: str, env: dict[str, str]) -> TestCommandResult:
    test_dir = TESTS_DIR / "dotnet" / lib
    if not test_dir.is_dir():
        return TestCommandResult(False, 0)
    proc = subprocess.run(["dotnet", "run"], cwd=test_dir, env=env)
    return TestCommandResult(True, proc.returncode)


def _dotnet_list_tests() -> list[str]:
    return _list_tests_from_matches(
        "dotnet",
        "*/*.csproj",
        lambda _path: "native",
        lambda path: path.parent.name,
    )


# ── No-op adapters ──────────────────────────────────────────────────


def _noop_install_dependencies(_lib: str, _ecosystem: str) -> None:
    pass


def _noop_prebuild(_lib: str) -> None:
    pass


# ── Test discovery ──────────────────────────────────────────────────


def _list_tests_from_matches(
    language: str,
    pattern: str,
    ecosystem_for_path: Callable[[Path], str],
    library_for_path: Callable[[Path], str] | None = None,
) -> list[str]:
    tests: list[str] = []
    for path in sorted((TESTS_DIR / language).glob(pattern)):
        library = library_for_path(path) if library_for_path is not None else path.parent.name
        ecosystem = ecosystem_for_path(path)
        tests.append(f"{language}-{library}-{ecosystem}")
    return tests


# ── Adapter registry ────────────────────────────────────────────────


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
    from genai_otel_conformance.results import split_test_name

    lang, lib, eco = split_test_name(name)
    adapter = LANGUAGE_ADAPTERS.get(lang)
    if adapter is None:
        return TestCommandResult(False, 0)
    return adapter.run_test(lib, eco, env)


def list_available_tests() -> list[str]:
    """Discover all available test names."""
    tests: list[str] = []
    for adapter in LANGUAGE_ADAPTERS.values():
        tests.extend(adapter.list_tests())
    return tests
