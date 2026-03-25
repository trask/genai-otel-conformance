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
from pathlib import Path
from typing import NamedTuple

from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.language_adapters import (
    LANGUAGE_ADAPTERS,
    install_with_uv,
    list_available_tests,
    run_test_cmd,
)
from genai_otel_conformance.statuses import (
    build_present_signal_entries,
    build_span_type_present_names,
)
from genai_otel_conformance.results import (
    TestResult,
    parse_result_dir,
    split_test_name,
)
from genai_otel_conformance.specs import (
    GENAI_EVENT_TYPES,
    GENAI_METRIC_TYPES,
    SPAN_TYPE_ORDER,
    SPAN_TYPE_SPECS,
)
from genai_otel_conformance.locations import TestLocation
from genai_otel_conformance.weaver import (
    SEMCONV_VERSION,
    ensure_weaver,
    path_from_env,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MOCK_SERVER_PORT = 8080


def _allocate_free_tcp_ports(count: int) -> list[int]:
    """Ask the OS for unused loopback TCP ports to reduce collisions in CI."""
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        ports: list[int] = []
        for sock in sockets:
            ports.append(int(sock.getsockname()[1]))
        return ports
    finally:
        for sock in sockets:
            sock.close()


GeneratedTestPayload = dict[str, object]


class GeneratedTestData(NamedTuple):
    path: Path
    data: GeneratedTestPayload
    has_relevant_data: bool


# ── Test data generation ────────────────────────────────────────────


def _data_path_from_test_name(test_name: str) -> Path:
    """Compute the data file path from a test name."""
    return TestLocation.from_test_name(test_name).data_file(TESTS_DIR)


def _results_dir_from_test_name(test_name: str) -> Path:
    """Compute the results directory path from a test name."""
    return TestLocation.from_test_name(test_name).results_dir(TESTS_DIR)


def _prepare_results_dir(result_dir: Path) -> None:
    """Ensure the result directory starts empty for a fresh Weaver run."""
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


def _has_weaver_output(result_dir: Path) -> bool:
    """Return whether Weaver wrote any JSON output files for this run."""
    return any(result_dir.glob("**/*.json"))


def _build_single_test_data(test_name: str, result: TestResult) -> GeneratedTestData:
    """Build committed dashboard data from a parsed Weaver result."""
    event_entries = build_present_signal_entries(
        GENAI_EVENT_TYPES,
        result.seen_events,
        result.detected.events,
    )
    metric_entries = build_present_signal_entries(
        GENAI_METRIC_TYPES,
        result.seen_metrics,
        result.detected.metrics,
    )
    has_genai_signals = bool(event_entries) or bool(metric_entries)
    spans = build_span_type_present_names(result, SPAN_TYPE_ORDER, SPAN_TYPE_SPECS)
    path = _data_path_from_test_name(test_name)

    data: GeneratedTestPayload = {
        "events": event_entries,
        "metrics": metric_entries,
    }
    if spans:
        data["spans"] = spans

    return GeneratedTestData(
        path=path,
        data=_normalize_generated_test_payload(data),
        has_relevant_data=bool(spans) or has_genai_signals,
    )


def _normalize_generated_test_payload(data: GeneratedTestPayload) -> GeneratedTestPayload:
    """Drop empty top-level objects and sort span attribute names alphabetically."""
    normalized: GeneratedTestPayload = {}
    for key, value in data.items():
        if not value:
            continue
        if key == "spans" and isinstance(value, dict):
            normalized[key] = {
                span_type: sorted(attrs)
                for span_type, attrs in value.items()
                if attrs
            }
            if not normalized[key]:
                normalized.pop(key)
            continue
        if key in ("events", "metrics") and isinstance(value, dict):
            normalized[key] = dict(sorted(value.items()))
            continue
        normalized[key] = value
    return normalized


def generate_single_test_data(test_name: str) -> GeneratedTestData | None:
    """Generate data for a single test from its results directory.

    Returns generated dashboard data or None if the Weaver output could not be parsed.
    """
    result_dir = _results_dir_from_test_name(test_name)
    result = parse_result_dir(result_dir, test_name)
    if result is None:
        return None
    return _build_single_test_data(test_name, result)



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


def _ensure_semconv_registry() -> str:
    registry = os.environ.get("REGISTRY")
    if registry:
        return registry

    semconv_cache_root = path_from_env(
        "SEMCONV_CACHE",
        Path.home() / ".cache" / "otel-conformance" / "semconv",
    )
    semconv_cache = semconv_cache_root / SEMCONV_VERSION.replace("/", "_")
    model_dir = semconv_cache / "model"

    if not model_dir.is_dir():
        print(f"=== Caching semantic conventions registry ({SEMCONV_VERSION}) ===")
        semconv_cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                SEMCONV_VERSION,
                "--depth",
                "1",
                "-q",
                "https://github.com/open-telemetry/semantic-conventions.git",
                str(semconv_cache),
            ],
            check=True,
        )

    return str(model_dir)


def _start_mock_server(mock_url: str) -> subprocess.Popen | None:
    health_url = f"{mock_url}/health"
    if is_healthy(health_url):
        print(f"Mock server already running on port {MOCK_SERVER_PORT}")
        return None

    install_with_uv("-e", "tests/mock-server", label="shared mock server dependencies")
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
    wait_for_health(health_url, 30, "Mock server", mock_proc)
    return mock_proc


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

    weaver_port, admin_port = _allocate_free_tcp_ports(2)
    registry = _ensure_semconv_registry()

    # ── Mock server ─────────────────────────────────────────────────

    mock_url = f"http://127.0.0.1:{MOCK_SERVER_PORT}"
    mock_proc: subprocess.Popen | None = None
    weaver_proc: subprocess.Popen | None = None

    try:
        mock_proc = _start_mock_server(mock_url)

        # ── Pre-build (compile before starting weaver) ──────────────
        # Weaver uses an inactivity timeout; long builds (e.g. Gradle)
        # can cause it to shut down before the test sends any data.

        LANGUAGE_ADAPTERS[lang].prebuild_test(lib)

        # ── Start weaver ────────────────────────────────────────────

        test_results_dir = _results_dir_from_test_name(test_name).resolve()
        _prepare_results_dir(test_results_dir)

        print(f"=== Starting weaver live-check for: {test_name} (ports {weaver_port}/{admin_port}) ===")
        weaver_bin = ensure_weaver()
        weaver_cmd = _build_weaver_command(
            weaver_bin,
            registry,
            test_results_dir,
            weaver_port,
            admin_port,
            extra_weaver_args,
        )

        weaver_proc = subprocess.Popen(weaver_cmd)

        print("Waiting for weaver to be ready...")
        wait_for_health(f"http://localhost:{admin_port}/health", 60, "Weaver", weaver_proc)

        test_env = _build_test_environment(mock_url, weaver_port)
        print(f"=== Running test: {test_name} ===")

        test_run = run_test_cmd(test_name, test_env)
        if not test_run.found:
            print(f"ERROR: Could not find test '{test_name}'", file=sys.stderr)
            _print_available_tests()
            sys.exit(1)

        # ── Stop weaver ─────────────────────────────────────────────

        weaver_exit = _stop_weaver(admin_port, weaver_proc)
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
        result = _write_generated_test_data(test_name)
        if result is None:
            print(f"ERROR: Could not parse Weaver results for test: {test_name}", file=sys.stderr)
            sys.exit(1)
        if not result.has_relevant_data:
            print(
                f"ERROR: No relevant data for test: {test_name}",
                file=sys.stderr,
            )
            sys.exit(1)

    finally:
        _stop_process(weaver_proc, "weaver")
        _stop_process(mock_proc, "mock server")


if __name__ == "__main__":
    main()
