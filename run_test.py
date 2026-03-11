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
from importlib import util as importlib_util
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

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

# Language directory names → display language names.
_LANG_DIRS = {"python": "Python", "java": "Java", "js": "JS", "dotnet": "C#"}


def _load_ecosystems() -> tuple[
    dict[str, str],
    dict[tuple[str, str], str],
]:
    """Load ecosystem definitions from tests/ecosystems.json.

    Returns (ecosystem_display, ecosystem_repos).
    """
    eco_file = TESTS_DIR / "ecosystems.json"
    if not eco_file.is_file():
        return {}, {}
    data = json.loads(eco_file.read_text(encoding="utf-8"))
    display: dict[str, str] = {}
    repos: dict[tuple[str, str], str] = {}
    for eco, info in data.items():
        display[eco] = info.get("display_name", eco)
        for lang_slug, repo in info.get("repos", {}).items():
            lang_display = _LANG_DIRS.get(lang_slug, lang_slug)
            repos[(eco, lang_display)] = repo
    return display, repos


ECOSYSTEM_DISPLAY, ECOSYSTEM_REPOS = _load_ecosystems()


# ── Data structures ──────────────────────────────────────────────────


class TestName(NamedTuple):
    language: str
    library: str
    ecosystem: str


class TestCommandResult(NamedTuple):
    found: bool
    exit_code: int


@dataclass
class TestResult:
    language: str
    library: str
    ecosystem: str
    statistics: dict | None
    violation_count: int
    violation_messages: list[str]
    entity_counts: dict[str, int]
    seen_attrs: dict[str, int]
    seen_non_registry_attrs: dict[str, int]
    seen_events: dict[str, int]
    seen_metrics: dict[str, int]
    has_data: bool
    detected_span_types: set[str] = field(default_factory=set)
    per_type_attrs: dict[str, set[str]] = field(default_factory=dict)
    detected_events: dict[str, int] = field(default_factory=dict)
    detected_metrics: dict[str, int] = field(default_factory=dict)


def parse_test_name(test_name: str) -> TestName:
    """Parse a supported test name into display values.

    Supported test names use the canonical format:
        <lang>-<library>-<ecosystem>

    Example:
        python-openai-openllmetry
    """
    lang, library, ecosystem = _parse_test_name(test_name)
    return TestName(_LANG_DIRS[lang], library, ecosystem)


# ── JSON parsing ─────────────────────────────────────────────────────


def try_parse_json(content):
    """Parse JSON content — handles single object, array, or JSONL."""
    objects = []

    # Single JSON object or array
    try:
        data = json.loads(content)
        if isinstance(data, list):
            objects.extend(data)
        elif isinstance(data, dict):
            objects.append(data)
        return objects
    except json.JSONDecodeError:
        pass

    # JSONL
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return objects


# ── Span type specifications (from gen-ai-spans.md and gen-ai-agent-spans.md) ─

# Expected attributes per span type, grouped by requirement level.
# Source: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
# Source: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-agent-spans.md

# Shared attribute lists to avoid repetition across span type specs.
_COMMON_REQUIRED = ["gen_ai.operation.name"]
_PROVIDER_REQUIRED = ["gen_ai.provider.name", "gen_ai.system"]
_COMMON_COND_REQUIRED = ["error.type"]
_CLIENT_COND_REQUIRED = ["gen_ai.request.model", "server.port"]
_CLIENT_RECOMMENDED = ["server.address"]
_INFERENCE_COND_REQUIRED = [
    "gen_ai.conversation.id",
    "gen_ai.output.type",
    "gen_ai.request.choice.count",
    "gen_ai.request.seed",
]
_INFERENCE_RECOMMENDED = [
    "gen_ai.request.frequency_penalty",
    "gen_ai.request.max_tokens",
    "gen_ai.request.presence_penalty",
    "gen_ai.request.stop_sequences",
    "gen_ai.request.temperature",
    "gen_ai.request.top_p",
    "gen_ai.response.finish_reasons",
    "gen_ai.response.id",
    "gen_ai.response.model",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
]

SPAN_TYPE_SPECS = {
    "inference": {
        "label": "Inference",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.response.finish_reasons", "gen_ai.response.id",
            "gen_ai.usage.output_tokens", "gen_ai.request.max_tokens",
            "gen_ai.request.temperature", "gen_ai.output.type",
            "gen_ai.usage.input_tokens",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INFERENCE_COND_REQUIRED,
        "recommended": _INFERENCE_RECOMMENDED + [
            "gen_ai.request.top_k",
        ] + _CLIENT_RECOMMENDED,
    },
    "embeddings": {
        "label": "Embeddings",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.embeddings.dimension.count", "gen_ai.request.encoding_formats",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED,
        "recommended": [
            "gen_ai.embeddings.dimension.count",
            "gen_ai.request.encoding_formats",
            "gen_ai.response.model",
            "gen_ai.usage.input_tokens",
        ] + _CLIENT_RECOMMENDED,
    },
    "retrieval": {
        "label": "Retrieval",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.data_source.id",
        },
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + [
            "gen_ai.data_source.id",
            "gen_ai.provider.name",
            "gen_ai.system",
        ] + _CLIENT_COND_REQUIRED,
        "recommended": [
            "gen_ai.request.top_k",
        ] + _CLIENT_RECOMMENDED,
    },
    "execute_tool": {
        "label": "Execute Tool",
        "expected_kind": "internal",
        "discriminator_attrs": {
            "gen_ai.tool.call.id", "gen_ai.tool.name", "gen_ai.tool.type",
        },
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED,
        "recommended": [
            "gen_ai.tool.call.id",
            "gen_ai.tool.description",
            "gen_ai.tool.name",
            "gen_ai.tool.type",
        ],
    },
    "create_agent": {
        "label": "Create Agent",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.agent.id", "gen_ai.agent.name",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.agent.version",
        ],
        "recommended": _CLIENT_RECOMMENDED,
    },
    "invoke_agent": {
        "label": "Invoke Agent",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.agent.id", "gen_ai.agent.name",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INFERENCE_COND_REQUIRED + [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.agent.version",
            "gen_ai.data_source.id",
        ],
        "recommended": _INFERENCE_RECOMMENDED + _CLIENT_RECOMMENDED,
    },
    "invoke_workflow": {
        "label": "Invoke Workflow",
        "expected_kind": "internal",
        "discriminator_attrs": {
            "gen_ai.workflow.name",
        },
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + [
            "gen_ai.workflow.name",
        ],
        "recommended": [],
    },
}

SPAN_TYPE_ORDER = ["create_agent", "invoke_agent", "invoke_workflow", "inference", "embeddings", "retrieval", "execute_tool"]


# ── Results parsing ──────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _load_test_metadata(lang: str, library: str) -> dict:
    """Load metadata.json for a test directory."""
    meta_file = TESTS_DIR / lang / library / "metadata.json"
    if not meta_file.is_file():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _version_package_from_metadata(lang: str, library: str, ecosystem: str) -> str:
    """Return the exact manifest package key used for version display."""
    metadata = _load_test_metadata(lang, library)
    version_packages = metadata.get("version_packages", {})
    if not isinstance(version_packages, dict):
        return ""
    package_name = version_packages.get(ecosystem, "")
    return package_name if isinstance(package_name, str) else ""


def _read_deps_from_test_dir(
    lang: str, library: str, ecosystem: str,
) -> dict[str, str]:
    """Read dependency versions from the test's dependency file.

    Returns a dict of {package_name: version}.
    """
    test_dir = TESTS_DIR / lang / library
    versions: dict[str, str] = {}

    if lang == "python":
        req_file = test_dir / f"requirements-{ecosystem}.txt"
        if req_file.exists():
            for line in req_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "==" in line:
                    pkg, ver = line.split("==", 1)
                    versions[pkg.strip()] = ver.strip()
    elif lang == "js":
        pkg_file = test_dir / "package.json"
        if pkg_file.exists():
            try:
                data = json.loads(pkg_file.read_text(encoding="utf-8"))
                versions = dict(data.get("dependencies", {}))
            except (OSError, json.JSONDecodeError):
                pass
    elif lang == "java":
        gradle_file = test_dir / "build.gradle.kts"
        if gradle_file.exists():
            content = gradle_file.read_text(encoding="utf-8")
            for m in re.finditer(r'implementation\("([^"]+)"\)', content):
                coord = m.group(1)
                parts = coord.rsplit(":", 1)
                if len(parts) == 2:
                    versions[parts[0]] = parts[1]
    elif lang == "dotnet":
        for csproj in test_dir.glob("*.csproj"):
            content = csproj.read_text(encoding="utf-8")
            for m in re.finditer(
                r'PackageReference\s+Include="([^"]+)"\s+Version="([^"]+)"',
                content,
            ):
                versions[m.group(1)] = m.group(2)

    return versions


def extract_version_from_deps(
    lang: str, library: str, ecosystem: str,
) -> str:
    """Extract the display version from checked-in dependency files.

    Reads requirements-*.txt (Python), package.json (JS), build.gradle.kts
    (Java), or *.csproj (.NET) and returns the version of the exact package
    named in metadata.json for this ecosystem, or ``""`` when not found.
    """
    versions = _read_deps_from_test_dir(lang, library, ecosystem)
    package_name = _version_package_from_metadata(lang, library, ecosystem)
    if not package_name:
        return ""
    return versions.get(package_name, "")


def _classify_span(span_name: str, span_attrs: dict[str, object]) -> set[str]:
    """Classify a span into span types using heuristics on individual span data.

    Returns a set of matching span type keys (e.g. {"embeddings", "inference"}).
    This enables detection of non-conforming spans that lack standard
    discriminator attributes.
    """
    types: set[str] = set()
    name_lower = span_name.lower()
    op_name = str(span_attrs.get("gen_ai.operation.name", "")).lower()
    oi_kind = str(span_attrs.get("openinference.span.kind", "")).upper()
    llm_type = str(span_attrs.get("llm.request.type", "")).lower()
    tl_kind = str(span_attrs.get("traceloop.span.kind", "")).lower()

    # ── Embeddings ────────────────────────────────────────────────
    if "embed" in name_lower:
        types.add("embeddings")
    elif span_attrs.get("embedding.model_name"):
        types.add("embeddings")
    elif oi_kind == "EMBEDDING":
        types.add("embeddings")
    elif llm_type in ("embedding", "embeddings"):
        types.add("embeddings")
    elif op_name in ("embedding", "embeddings"):
        types.add("embeddings")

    # ── Inference (chat / completion) ─────────────────────────────
    if op_name == "chat":
        types.add("inference")
    elif oi_kind == "LLM":
        types.add("inference")
    elif llm_type in ("chat", "completion"):
        types.add("inference")
    elif op_name == "generate_content":
        types.add("inference")
    # js-vercel-ai-native: spans have gen_ai.response.* and gen_ai.usage.*
    # but no gen_ai.operation.name
    elif span_attrs.get("gen_ai.usage.output_tokens") is not None \
            and span_attrs.get("gen_ai.response.finish_reasons") is not None:
        types.add("inference")
    # promptflow-native: openai_chat spans use llm.* attributes
    elif span_attrs.get("llm.response.model") is not None \
            and span_attrs.get("llm.usage.completion_tokens") is not None:
        types.add("inference")

    # ── Create Agent ──────────────────────────────────────────────
    if op_name == "create_agent":
        types.add("create_agent")

    # ── Invoke Agent ──────────────────────────────────────────────
    if oi_kind == "AGENT":
        types.add("invoke_agent")
    elif op_name == "invoke_agent":
        types.add("invoke_agent")
    elif span_attrs.get("gen_ai.agent.name") or span_attrs.get("gen_ai.agent.id"):
        if op_name != "create_agent":
            types.add("invoke_agent")
    elif span_attrs.get("crewai.agent.id") or span_attrs.get("crewai.agent.role"):
        types.add("invoke_agent")

    # ── Execute Tool ──────────────────────────────────────────────
    if op_name == "execute_tool":
        types.add("execute_tool")
    elif oi_kind == "TOOL":
        types.add("execute_tool")
    elif span_attrs.get("gen_ai.tool.name") or span_attrs.get("gen_ai.tool.call.id"):
        types.add("execute_tool")

    # ── Invoke Workflow ───────────────────────────────────────────
    if op_name == "invoke_workflow":
        types.add("invoke_workflow")
    elif span_attrs.get("traceloop.workflow.name"):
        types.add("invoke_workflow")
    elif name_lower == "crewai.workflow":
        types.add("invoke_workflow")
    elif span_attrs.get("crewai.crew.id"):
        types.add("invoke_workflow")

    # ── Retrieval ─────────────────────────────────────────────────
    if op_name == "retrieval":
        types.add("retrieval")
    elif oi_kind == "RETRIEVER":
        types.add("retrieval")
    elif span_attrs.get("gen_ai.data_source.id"):
        types.add("retrieval")

    return types


def _extract_span_types_from_samples(
    all_objects: list[dict],
) -> tuple[set[str], dict[str, set[str]]]:
    """Scan Weaver sample spans, classify each one, and track per-type attrs.

    Returns (detected_span_types, per_type_attrs) where per_type_attrs maps
    each span-type key to the set of attribute names present on spans of that type.
    """
    span_types: set[str] = set()
    per_type_attrs: dict[str, set[str]] = {}
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        samples = obj.get("samples", [])
        for sample in samples:
            span = sample.get("span")
            if not span:
                continue
            span_name = span.get("name", "")
            # Build a simple attr-name -> value dict from the span's attributes.
            attrs: dict[str, object] = {}
            for attr in span.get("attributes", []):
                attrs[attr.get("name", "")] = attr.get("value")
            classified = _classify_span(span_name, attrs)
            span_types |= classified
            attr_names = set(attrs.keys())
            for st in classified:
                if st not in per_type_attrs:
                    per_type_attrs[st] = set()
                per_type_attrs[st] |= attr_names
    return span_types, per_type_attrs


def _extract_events_from_samples(all_objects: list[dict]) -> dict[str, int]:
    """Extract GenAI event names from log records in samples."""
    events: dict[str, int] = {}
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            log = sample.get("log")
            if not log:
                continue
            event_name = log.get("event_name", "")
            if event_name.startswith("gen_ai."):
                events[event_name] = events.get(event_name, 0) + 1
    return events


def _extract_metrics_from_samples(all_objects: list[dict]) -> dict[str, int]:
    """Extract GenAI metric names from metric records in samples."""
    metrics: dict[str, int] = {}
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            metric = sample.get("metric")
            if not metric:
                continue
            metric_name = metric.get("name", "")
            if metric_name.startswith("gen_ai."):
                metrics[metric_name] = metrics.get(metric_name, 0) + 1
    return metrics


def parse_result_dir(result_dir: Path, test_name: str) -> TestResult | None:
    """Parse a single test's Weaver output directory into a TestResult."""
    if not result_dir.is_dir():
        return None

    all_objects = []

    for json_file in sorted(result_dir.glob("**/*.json")):
        try:
            content = json_file.read_text(encoding="utf-8")
            all_objects.extend(try_parse_json(content))
        except (OSError, ValueError) as e:
            print(f"Warning: Could not parse {json_file}: {e}", file=sys.stderr)

    statistics = None
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        if "statistics" in obj and isinstance(obj["statistics"], dict):
            statistics = obj["statistics"]
        elif "registry_coverage" in obj or "advice_level_counts" in obj:
            statistics = obj

    # Extract seen_registry_attributes with non-zero counts
    seen_attrs = {}
    if statistics:
        for attr, count in statistics.get("seen_registry_attributes", {}).items():
            if count > 0:
                seen_attrs[attr] = count

    # Extract seen non-registry attributes with non-zero counts
    seen_non_registry_attrs = {}
    if statistics:
        for attr, count in statistics.get("seen_non_registry_attributes", {}).items():
            if count > 0:
                seen_non_registry_attrs[attr] = count

    # Extract seen events (registry and non-registry)
    seen_events = {}
    if statistics:
        for attr, count in statistics.get("seen_registry_events", {}).items():
            if count > 0:
                seen_events[attr] = count
        for attr, count in statistics.get("seen_non_registry_events", {}).items():
            if count > 0:
                seen_events[attr] = count

    # Extract seen metrics (registry and non-registry)
    seen_metrics = {}
    if statistics:
        for attr, count in statistics.get("seen_registry_metrics", {}).items():
            if count > 0:
                seen_metrics[attr] = count
        for attr, count in statistics.get("seen_non_registry_metrics", {}).items():
            if count > 0:
                seen_metrics[attr] = count

    # Violation count from weaver's authoritative advice_level_counts
    violation_count = 0
    if statistics:
        violation_count = statistics.get("advice_level_counts", {}).get("violation", 0)

    # Collect distinct advisory messages (filtered: skip "not stable").
    # Uses advice_message_counts from statistics for completeness.
    violation_messages: set[str] = set()
    if statistics:
        for msg in statistics.get("advice_message_counts", {}):
            if "not stable" not in msg.lower():
                violation_messages.add(msg)

    # Entity counts from statistics
    entity_counts = {}
    if statistics:
        entity_counts = statistics.get("total_entities_by_type", {})

    try:
        language, library, ecosystem = parse_test_name(test_name)
    except ValueError:
        print(f"Warning: Could not parse test name: {test_name}", file=sys.stderr)
        return None

    has_data = bool(statistics and statistics.get("total_entities", 0) > 0)

    # Classify individual spans from samples for span-type detection.
    detected_span_types, per_type_attrs = _extract_span_types_from_samples(all_objects)

    # Extract GenAI events and metrics from log/metric samples.
    detected_events = _extract_events_from_samples(all_objects)
    detected_metrics = _extract_metrics_from_samples(all_objects)

    # Also pick up gen_ai events from non-registry event statistics
    # (Weaver may not include them in the registry yet).
    if statistics:
        for ev_name, count in statistics.get("seen_non_registry_events", {}).items():
            if count > 0 and ev_name.startswith("gen_ai."):
                detected_events[ev_name] = max(detected_events.get(ev_name, 0), count)

        for metric_name, count in statistics.get("seen_non_registry_metrics", {}).items():
            if count > 0 and metric_name.startswith("gen_ai."):
                detected_metrics[metric_name] = max(detected_metrics.get(metric_name, 0), count)

    return TestResult(
        language=language,
        library=library,
        ecosystem=ecosystem,
        statistics=statistics,
        violation_count=violation_count,
        violation_messages=sorted(violation_messages),
        entity_counts=entity_counts,
        seen_attrs=seen_attrs,
        seen_non_registry_attrs=seen_non_registry_attrs,
        seen_events=seen_events,
        seen_metrics=seen_metrics,
        has_data=has_data,
        detected_span_types=detected_span_types,
        per_type_attrs=per_type_attrs,
        detected_events=detected_events,
        detected_metrics=detected_metrics,
    )


# ── Test name parsing (internal three-part split) ───────────────────


def _parse_test_name(name: str) -> tuple[str, str, str]:
    """Parse a test name into language/library/ecosystem slugs."""
    try:
        lang, rest = name.split("-", 1)
        lib, eco = rest.rsplit("-", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid test name: {name}") from exc

    if lang not in _LANG_DIRS or not lib or not eco:
        raise ValueError(f"Invalid test name: {name}")

    return lang, lib, eco


# ── Test data generation ────────────────────────────────────────────


def _data_path_from_test_name(test_name: str) -> Path:
    """Compute the data file path from a test name."""
    lang, lib, eco = _parse_test_name(test_name)
    return TESTS_DIR / lang / lib / f"data-{eco}.json"


def _results_dir_from_test_name(test_name: str) -> Path:
    """Compute the results directory path from a test name."""
    lang, lib, eco = _parse_test_name(test_name)
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

    all_present = set(r.seen_attrs) | set(r.seen_non_registry_attrs)
    has_genai = any(a.startswith("gen_ai.") for a in all_present)
    if not has_genai and not r.detected_span_types:
        return None

    # Compute span-type attribute statuses.
    span_types: dict[str, dict[str, str]] = {}
    for st_key in SPAN_TYPE_ORDER:
        spec = SPAN_TYPE_SPECS[st_key]

        attr_names: list[str] = []
        for level in ("required", "conditionally_required", "recommended"):
            attr_names.extend(spec.get(level, []))
        if not attr_names:
            continue

        all_spec_attrs: set[str] = set(attr_names)
        discriminators = spec.get("discriminator_attrs", set())
        is_relevant = False
        if discriminators:
            is_relevant = bool(all_present & discriminators) or st_key in r.detected_span_types
        else:
            is_relevant = bool(all_present & all_spec_attrs)

        if not is_relevant:
            continue

        # Use per-span-type attribute presence when available;
        # fall back to global presence for span types not detected
        # via sample classification (e.g. when only Weaver stats exist).
        type_present = r.per_type_attrs.get(st_key, all_present)
        span_types[st_key] = {
            attr: ("present" if attr in type_present else "absent")
            for attr in attr_names
        }

    if not span_types:
        return None

    path = _data_path_from_test_name(test_name)

    data = {
        "span_types": span_types,
    }

    return path, data


# ── Test discovery and execution ────────────────────────────────────


def _gradle_cmd(test_dir: Path) -> list[str]:
    """Return the Gradle wrapper command for the given test directory."""
    if sys.platform == "win32":
        return [str((test_dir / "gradlew.bat").resolve())]
    return ["./gradlew"]


def _ensure_python_test_support() -> None:
    """Ensure the shared Python test support package is importable."""
    if importlib_util.find_spec("otel_setup") is not None:
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(TESTS_DIR / "python")],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print("ERROR: Failed to install shared Python test support.", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        sys.exit(exc.returncode or 1)


def run_test_cmd(name: str, env: dict[str, str]) -> TestCommandResult:
    """Run the test command.

    Returns whether the test was found and the test command's exit code.
    Weaver violations are handled separately from test-process failures.
    """
    lang, lib, eco = _parse_test_name(name)

    if lang == "python":
        _ensure_python_test_support()
        test_file = Path(f"tests/python/{lib}/test_{eco}.py")
        if not test_file.is_file():
            return TestCommandResult(False, 0)
        proc = subprocess.run([sys.executable, str(test_file)], env=env)
        return TestCommandResult(True, proc.returncode)
    elif lang == "js":
        test_dir = Path(f"tests/js/{lib}")
        if not test_dir.is_dir():
            return TestCommandResult(False, 0)
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        install_proc = subprocess.run([npm, "install", "--silent"], cwd=test_dir, env=env)
        if install_proc.returncode != 0:
            return TestCommandResult(True, install_proc.returncode)
        test_proc = subprocess.run([npm, "run", f"test:{eco}"], cwd=test_dir, env=env)
        return TestCommandResult(True, test_proc.returncode)
    elif lang == "java":
        test_dir = Path(f"tests/java/{lib}")
        if not test_dir.is_dir():
            return TestCommandResult(False, 0)
        gradle = _gradle_cmd(test_dir)
        build_file = test_dir / "build.gradle.kts"
        gradle_task = "bootRun" if "spring-boot" in build_file.read_text() else "run"
        proc = subprocess.run([*gradle, gradle_task], cwd=test_dir, env=env)
        return TestCommandResult(True, proc.returncode)
    elif lang == "dotnet":
        test_dir = Path(f"tests/dotnet/{lib}")
        if not test_dir.is_dir():
            return TestCommandResult(False, 0)
        proc = subprocess.run(["dotnet", "run"], cwd=test_dir, env=env)
        return TestCommandResult(True, proc.returncode)
    else:
        return TestCommandResult(False, 0)


def list_available_tests() -> list[str]:
    """Discover all available test names."""
    tests: list[str] = []
    for f in sorted(Path("tests/python").glob("*/test_*.py")):
        lib = f.parent.name
        eco = f.stem.removeprefix("test_")
        tests.append(f"python-{lib}-{eco}")
    for f in sorted(Path("tests/js").glob("*/test_*.ts")):
        lib = f.parent.name
        eco = f.stem.removeprefix("test_")
        tests.append(f"js-{lib}-{eco}")
    for f in sorted(Path("tests/java").glob("*/build.gradle.kts")):
        lib = f.parent.name
        if "spring-boot" in f.read_text():
            tests.append(f"java-{lib}-native")
        else:
            tests.append(f"java-{lib}-otelcontrib")
    for f in sorted(Path("tests/dotnet").glob("*/*.csproj")):
        lib = f.parent.name
        tests.append(f"dotnet-{lib}-native")
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
        lang, lib, eco = _parse_test_name(test_name)
    except ValueError:
        print(f"ERROR: Invalid test name '{test_name}'", file=sys.stderr)
        print("Expected format: <lang>-<library>-<ecosystem>", file=sys.stderr)
        _print_available_tests()
        sys.exit(1)

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
            print(f"=== Starting mock server on port {MOCK_SERVER_PORT} ===")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-e",
                     str(SCRIPT_DIR / "tests" / "mock-server")],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                print("ERROR: Failed to install mock server package.", file=sys.stderr)
                if exc.stdout:
                    print(exc.stdout, file=sys.stderr, end="")
                if exc.stderr:
                    print(exc.stderr, file=sys.stderr, end="")
                sys.exit(exc.returncode or 1)
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

        if lang == "java":
            test_dir = Path(f"tests/java/{lib}")
            gradle = _gradle_cmd(test_dir)
            print(f"=== Pre-building Java project in {test_dir} ===")
            subprocess.run([*gradle, "classes"], cwd=test_dir, check=True)
        elif lang == "dotnet":
            test_dir = Path(f"tests/dotnet/{lib}")
            print(f"=== Pre-building .NET project in {test_dir} ===")
            subprocess.run(["dotnet", "build"], cwd=test_dir, check=True)

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
