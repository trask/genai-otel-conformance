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
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.language_adapters import (
    LANGUAGE_ADAPTERS,
    UvNotInstalledError,
    install_with_uv,
    list_available_tests,
    run_test_cmd,
)
from genai_otel_conformance.locations import TestLocation
from genai_otel_conformance.results import TestResult, parse_result_dir
from genai_otel_conformance.data_files import (
    GeneratedTestData,
    generate_single_test_data,
)
from genai_otel_conformance.weaver import (
    ensure_semconv_registry,
    ensure_weaver,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MOCK_SERVER_PORT = 8080


@dataclass(frozen=True)
class RunTestError(Exception):
    message: str
    exit_code: int = 1
    show_available_tests: bool = False

    def __str__(self) -> str:
        return self.message


@dataclass
class PipelineState:
    mock_proc: subprocess.Popen | None = None
    weaver_proc: subprocess.Popen | None = None


def _allocate_free_tcp_ports(count: int) -> list[int]:
    """Ask the OS for unused loopback TCP ports to reduce collisions in CI."""
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _prepare_results_dir(result_dir: Path) -> None:
    """Ensure the result directory starts empty for a fresh Weaver run."""
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


def _has_weaver_output(result_dir: Path) -> bool:
    """Return whether Weaver wrote any JSON output files for this run."""
    return any(result_dir.glob("**/*.json"))


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
            raise RunTestError(f"{label} process died during startup")
        time.sleep(1)
    raise RunTestError(f"{label} failed to become ready after {timeout}s")


def _start_mock_server(mock_url: str, state: PipelineState) -> None:
    health_url = f"{mock_url}/health"
    if is_healthy(health_url):
        print(f"Mock server already running on port {MOCK_SERVER_PORT}")
        return

    install_with_uv("-e", "tests/mock-server", label="shared mock server dependencies")
    print(f"=== Starting mock server on port {MOCK_SERVER_PORT} ===")
    state.mock_proc = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT_DIR / "tests" / "mock-server" / "mock_server" / "server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(MOCK_SERVER_PORT),
        ],
    )
    wait_for_health(health_url, 30, "Mock server", state.mock_proc)


def _build_weaver_command(
    weaver_bin: Path,
    registry: str,
    result_dir: Path,
    weaver_port: int,
    admin_port: int,
    extra_weaver_args: list[str],
) -> list[str]:
    command = [str(weaver_bin), "registry", "live-check"]
    if registry:
        command.extend(["-r", registry])
    command.extend(
        [
            "--format",
            "json",
            "--output",
            str(result_dir),
            "--otlp-grpc-port",
            str(weaver_port),
            "--admin-port",
            str(admin_port),
            "--inactivity-timeout",
            "30",
        ]
    )
    command.extend(extra_weaver_args)
    return command


def _stop_weaver(admin_port: int, weaver_proc: subprocess.Popen) -> int:
    time.sleep(1)
    if weaver_proc.poll() is None:
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"http://localhost:{admin_port}/stop", method="POST"),
                timeout=5,
            )
        except Exception:
            weaver_proc.terminate()
    return weaver_proc.wait()


def _build_test_environment(mock_url: str, weaver_port: int) -> dict[str, str]:
    return {
        **os.environ,
        "MOCK_LLM_URL": mock_url,
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{weaver_port}",
    }


def _write_generated_test_data(test_name: str) -> GeneratedTestData | None:
    result = generate_single_test_data(test_name)
    if result is None:
        return None

    result.path.parent.mkdir(parents=True, exist_ok=True)
    result.path.write_text(json.dumps(result.data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {result.path}")
    return result


def _stop_process(proc: subprocess.Popen | None, label: str) -> None:
    if proc and proc.poll() is None:
        print(f"Stopping {label} (PID {proc.pid})...")
        proc.terminate()
        proc.wait()


def _print_available_tests() -> None:
    """Print the current list of runnable tests to stderr."""
    print("Available tests:", file=sys.stderr)
    for test_name in list_available_tests():
        print(f"  {test_name}", file=sys.stderr)


# ── Main ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    test_name: str
    extra_weaver_args: list[str]
    mock_url: str
    weaver_port: int
    admin_port: int
    registry: str


def _start_weaver_live_check(
    config: PipelineConfig,
    state: PipelineState,
    location: TestLocation,
) -> Path:
    """Start Weaver live-check and return the result directory."""
    test_results_dir = location.results_dir(TESTS_DIR).resolve()
    _prepare_results_dir(test_results_dir)

    print(f"=== Starting weaver live-check for: {config.test_name} (ports {config.weaver_port}/{config.admin_port}) ===")
    weaver_bin = ensure_weaver()
    weaver_cmd = _build_weaver_command(
        weaver_bin,
        config.registry,
        test_results_dir,
        config.weaver_port,
        config.admin_port,
        config.extra_weaver_args,
    )

    state.weaver_proc = subprocess.Popen(weaver_cmd)

    print("Waiting for weaver to be ready...")
    wait_for_health(f"http://localhost:{config.admin_port}/health", 60, "Weaver", state.weaver_proc)
    return test_results_dir


def _finish_weaver_run(
    config: PipelineConfig,
    state: PipelineState,
    test_results_dir: Path,
) -> tuple[int, TestResult | None]:
    """Stop Weaver and load the fresh parsed result, if any."""
    if state.weaver_proc is None:
        raise RunTestError("Weaver process was not started")

    weaver_exit = _stop_weaver(config.admin_port, state.weaver_proc)
    state.weaver_proc = None
    print(f"=== Weaver exit code: {weaver_exit} ===")
    print(f"=== Results in: {test_results_dir} ===")
    return weaver_exit, parse_result_dir(test_results_dir, config.test_name)


def _validate_weaver_output(
    config: PipelineConfig,
    test_results_dir: Path,
    weaver_exit: int,
    fresh_result: TestResult | None,
) -> None:
    """Raise if Weaver output is missing or unusable."""
    has_weaver_output = _has_weaver_output(test_results_dir)
    has_weaver_stats = fresh_result is not None and fresh_result.statistics is not None

    if not has_weaver_output:
        raise RunTestError(
            f"Weaver produced no JSON output for test: {config.test_name}",
        )

    if weaver_exit != 0 and not has_weaver_stats:
        raise RunTestError(
            "Weaver exited non-zero before writing statistics.",
            exit_code=weaver_exit or 1,
        )
    if weaver_exit != 0:
        print(
            "Note: Weaver returned a non-zero exit code because violations were reported; continuing with captured statistics.",
            file=sys.stderr,
        )


def _run_test_pipeline(
    config: PipelineConfig,
    state: PipelineState,
) -> None:
    """Run the test pipeline: mock server, pre-build, weaver, test, collect results."""
    location = TestLocation.from_test_name(config.test_name)

    if state.mock_proc is None:
        _start_mock_server(config.mock_url, state)

    # Weaver uses an inactivity timeout; long builds (e.g. Gradle)
    # can cause it to shut down before the test sends any data.
    LANGUAGE_ADAPTERS[location.lang].prebuild_test(location.library)

    test_results_dir = _start_weaver_live_check(config, state, location)

    test_env = _build_test_environment(config.mock_url, config.weaver_port)
    print(f"=== Running test: {config.test_name} ===")
    test_run = run_test_cmd(config.test_name, test_env)
    if not test_run.found:
        raise RunTestError(
            f"Could not find test '{config.test_name}'",
            show_available_tests=True,
        )

    weaver_exit, fresh_result = _finish_weaver_run(config, state, test_results_dir)

    if test_run.exit_code != 0:
        raise RunTestError(
            f"Test command exited with code {test_run.exit_code}.",
            exit_code=test_run.exit_code or 1,
        )

    _validate_weaver_output(config, test_results_dir, weaver_exit, fresh_result)

    print("=== Updating test data file ===")
    result = _write_generated_test_data(config.test_name)
    if result is None:
        raise RunTestError(f"Could not parse Weaver results for test: {config.test_name}")
    if not result.has_relevant_data:
        raise RunTestError(f"No relevant data for test: {config.test_name}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python run_test.py <test-name> [weaver-args...]", file=sys.stderr)
        return 1

    test_name = sys.argv[1]
    extra_weaver_args = sys.argv[2:]

    print(f"Test: {test_name}")

    try:
        location = TestLocation.from_test_name(test_name)
    except ValueError:
        print(f"ERROR: Invalid test name '{test_name}'", file=sys.stderr)
        print("Expected format: <lang>-<library>-<ecosystem>", file=sys.stderr)
        _print_available_tests()
        return 1

    state = PipelineState()

    try:
        lang, lib = location.lang, location.library
        LANGUAGE_ADAPTERS[lang].install_dependencies(lib, location.ecosystem)

        weaver_port, admin_port = _allocate_free_tcp_ports(2)
        registry = ensure_semconv_registry()
        mock_url = f"http://127.0.0.1:{MOCK_SERVER_PORT}"

        config = PipelineConfig(
            test_name=test_name,
            extra_weaver_args=extra_weaver_args,
            mock_url=mock_url,
            weaver_port=weaver_port,
            admin_port=admin_port,
            registry=registry,
        )

        _run_test_pipeline(config, state)
    except RunTestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if exc.show_available_tests:
            _print_available_tests()
        return exc.exit_code
    except UvNotInstalledError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        _stop_process(state.weaver_proc, "weaver")
        _stop_process(state.mock_proc, "mock server")

    return 0


if __name__ == "__main__":
    sys.exit(main())
