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
    - weaver on PATH (https://github.com/open-telemetry/weaver)
    - Python 3.12+ (for mock server)
    - Language-specific toolchain for the test being run
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR / "tests"

# ── Language / ecosystem configuration ──────────────────────────────

# Language directory names → display language names.
_LANG_DIRS = {"python": "Python", "java": "Java", "js": "JS", "dotnet": "C#"}


def _load_ecosystems() -> tuple[
    list[str],
    dict[str, str],
    dict[tuple[str, str], str],
]:
    """Load ecosystem definitions from tests/ecosystems.json.

    Returns (ecosystem_suffixes, ecosystem_display, ecosystem_repos).
    """
    eco_file = TESTS_DIR / "ecosystems.json"
    if not eco_file.is_file():
        return [], {}, {}
    data = json.loads(eco_file.read_text(encoding="utf-8"))
    suffixes = list(data.keys())
    display: dict[str, str] = {}
    repos: dict[tuple[str, str], str] = {}
    for eco, info in data.items():
        display[eco] = info.get("display_name", eco)
        for lang_slug, repo in info.get("repos", {}).items():
            lang_display = _LANG_DIRS.get(lang_slug, lang_slug)
            repos[(eco, lang_display)] = repo
    return suffixes, display, repos


ECOSYSTEM_SUFFIXES, ECOSYSTEM_DISPLAY, ECOSYSTEM_REPOS = _load_ecosystems()

# New-style test name language prefixes (e.g. "python-cohere-openllmetry").
LANG_PREFIXES = {
    "python-": "Python",
    "js-": "JS",
    "java-": "Java",
    "dotnet-": "C#",
}


# ── Data structures ──────────────────────────────────────────────────


class TestName(NamedTuple):
    language: str
    library: str
    ecosystem: str


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
    has_data: bool
    versions: dict[str, str]
    detected_span_types: set[str] = field(default_factory=set)
    per_type_attrs: dict[str, set[str]] = field(default_factory=dict)
    detected_events: set[str] = field(default_factory=set)
    detected_metrics: set[str] = field(default_factory=set)


def parse_test_name(test_name) -> TestName:
    """Parse a test directory name into its components.

    Test names follow: <library>-<ecosystem>[_<language>]
    e.g. openai-otelcontrib, openai-otelcontrib_java, openai-openllmetry_js

    New-style names may use a language prefix instead:
    e.g. python-autogen-agentchat-openinference, js-anthropic-openinference

    The language suffix/prefix is optional; Python is the default when omitted.
    """
    lang_suffixes = {
        "_java": "Java",
        "_js": "JS",
        "_dotnet": "C#",
    }

    # Extract language suffix if present (old-style naming)
    language = "Python"
    base = test_name
    for suffix, lang_display in lang_suffixes.items():
        if test_name.endswith(suffix):
            language = lang_display
            base = test_name[: -len(suffix)]
            break

    # Extract language prefix if present (new-style naming)
    if language == "Python":
        for prefix, lang_display in LANG_PREFIXES.items():
            if base.startswith(prefix):
                language = lang_display
                base = base[len(prefix):]
                break

    # Try each known ecosystem suffix (longest first)
    for eco_suffix in sorted(ECOSYSTEM_SUFFIXES, key=len, reverse=True):
        if base.endswith(f"-{eco_suffix}"):
            library = base[: -(len(eco_suffix) + 1)]
            return TestName(language, library, eco_suffix)

    # Fallback: split on last dash
    parts = base.rsplit("-", 1)
    if len(parts) == 2:
        return TestName(language, parts[0], parts[1])

    return TestName(language, base, "unknown")


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

# Prefix patterns used to identify the instrumentation package per ecosystem.
_INSTRUMENTATION_PREFIXES = {
    "otelcontrib": [
        "opentelemetry-instrumentation-",       # Python
        "@opentelemetry/instrumentation-",       # JS
    ],
    "openllmetry": [
        "@traceloop/instrumentation-",           # JS
        "opentelemetry-instrumentation-",        # Python (traceloop's packages)
    ],
    "openinference": [
        "openinference-instrumentation-",        # Python
        "@arizeai/openinference-instrumentation-",  # JS
    ],
}

# Packages to skip when searching for the "main" library in native tests.
_INFRA_PREFIXES = (
    "opentelemetry", "@opentelemetry/", "otel", "tsx", "typescript",
    "grpc", "protobuf", "@grpc/",
)


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
    """Extract the instrumentation version from checked-in dependency files.

    Reads requirements-*.txt (Python), package.json (JS), build.gradle.kts
    (Java), or *.csproj (.NET) and returns the version of the instrumentation
    package, or ``""`` when not found.
    """
    versions = _read_deps_from_test_dir(lang, library, ecosystem)
    instr_version = _extract_instrumentation_version(ecosystem, library, versions)
    if instr_version:
        parts = instr_version.rsplit(" ", 1)
        return parts[-1] if len(parts) == 2 else instr_version
    return ""


def _extract_instrumentation_version(
    ecosystem: str, library: str, versions: dict[str, str],
) -> str:
    """Pick the most relevant instrumentation library version string.

    Returns a display string like ``"opentelemetry-instrumentation-openai-v2 2.0.0"``
    or ``""`` when no match is found.
    """
    if not versions:
        return ""

    # For instrumentation ecosystems, look for the instrumentation package.
    prefixes = _INSTRUMENTATION_PREFIXES.get(ecosystem)
    if prefixes:
        for prefix in prefixes:
            candidates = [
                (pkg, ver) for pkg, ver in versions.items()
                if pkg.lower().startswith(prefix.lower())
            ]
            if candidates:
                # Prefer the most specific (longest) package name.
                candidates.sort(key=lambda x: len(x[0]), reverse=True)
                return f"{candidates[0][0]} {candidates[0][1]}"

    # For Java otelcontrib, the maven coordinate contains "instrumentation".
    if ecosystem == "otelcontrib":
        for pkg, ver in versions.items():
            if "instrumentation" in pkg.lower() and ":" in pkg:
                return f"{pkg} {ver}"

    # For native tests, try to match the library name against package names.
    if ecosystem == "native":
        lib_norm = re.sub(r"[-_.]", "", library.lower())
        lib_tokens = [token for token in re.split(r"[-_.]+", library.lower()) if token]
        best = None
        for pkg, ver in versions.items():
            pkg_lower = re.sub(r"[-_.]", "", pkg.lower())
            if any(pkg_lower.startswith(ip) for ip in _INFRA_PREFIXES):
                continue
            token_match = lib_tokens and all(token in pkg_lower for token in lib_tokens)
            if lib_norm in pkg_lower or pkg_lower in lib_norm or token_match:
                if best is None or len(pkg) > len(best[0]):
                    best = (pkg, ver)
        if best:
            return f"{best[0]} {best[1]}"

    return ""


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


def _extract_events_from_samples(all_objects: list[dict]) -> set[str]:
    """Extract GenAI event names from log records in samples."""
    events: set[str] = set()
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            log = sample.get("log")
            if not log:
                continue
            event_name = log.get("event_name", "")
            if event_name.startswith("gen_ai."):
                events.add(event_name)
    return events


def _extract_metrics_from_samples(all_objects: list[dict]) -> set[str]:
    """Extract GenAI metric names from metric records in samples."""
    metrics: set[str] = set()
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            metric = sample.get("metric")
            if not metric:
                continue
            metric_name = metric.get("name", "")
            if metric_name.startswith("gen_ai."):
                metrics.add(metric_name)
    return metrics


def parse_result_dir(result_dir: Path, test_name: str) -> TestResult | None:
    """Parse a single test's Weaver output directory into a TestResult."""
    if not result_dir.is_dir():
        return None

    all_objects = []

    for json_file in sorted(result_dir.glob("**/*.json")):
        if json_file.name == "versions.json":
            continue
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

    language, library, ecosystem = parse_test_name(test_name)

    has_data = bool(statistics and statistics.get("total_entities", 0) > 0)

    # Read captured dependency versions
    versions = {}
    versions_file = result_dir / "versions.json"
    if versions_file.exists():
        try:
            versions = json.loads(versions_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

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
                detected_events.add(ev_name)

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
        has_data=has_data,
        versions=versions,
        detected_span_types=detected_span_types,
        per_type_attrs=per_type_attrs,
        detected_events=detected_events,
        detected_metrics=detected_metrics,
    )


# ── Test name parsing (internal three-part split) ───────────────────


def _parse_test_name(name: str) -> tuple[str, str, str]:
    lang, rest = name.split("-", 1)
    lib, eco = rest.rsplit("-", 1)
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


def run_test_cmd(name: str, env: dict[str, str]) -> bool:
    """Run the test command. Returns True if the test was found, False otherwise.

    Exit code is ignored — weaver is the arbiter of pass/fail.
    """
    lang, lib, eco = _parse_test_name(name)

    if lang == "python":
        test_file = Path(f"tests/python/{lib}/test_{eco}.py")
        if not test_file.is_file():
            return False
        subprocess.run([sys.executable, str(test_file)], env=env)
    elif lang == "js":
        test_dir = Path(f"tests/js/{lib}")
        if not test_dir.is_dir():
            return False
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.run([npm, "install", "--silent"], cwd=test_dir, env=env)
        subprocess.run([npm, "run", f"test:{eco}"], cwd=test_dir, env=env)
    elif lang == "java":
        test_dir = Path(f"tests/java/{lib}")
        if not test_dir.is_dir():
            return False
        gradle = _gradle_cmd(test_dir)
        build_file = test_dir / "build.gradle.kts"
        gradle_task = "bootRun" if "spring-boot" in build_file.read_text() else "run"
        subprocess.run([*gradle, gradle_task], cwd=test_dir, env=env)
    elif lang == "dotnet":
        test_dir = Path(f"tests/dotnet/{lib}")
        if not test_dir.is_dir():
            return False
        subprocess.run(["dotnet", "run"], cwd=test_dir, env=env)
    else:
        return False
    return True


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

    lang, lib, eco = _parse_test_name(test_name)
    mock_port = int(os.environ.get("MOCK_PORT", "8080"))

    weaver_port = random.randint(10000, 60000) & ~1  # even base
    admin_port = weaver_port + 1

    # ── Local registry cache ────────────────────────────────────────

    semconv_cache = Path(os.environ.get(
        "SEMCONV_CACHE",
        Path.home() / ".cache" / "otel-conformance" / "semconv",
    ))
    registry = os.environ.get("REGISTRY")
    if not registry:
        model_dir = semconv_cache / "model"
        if model_dir.is_dir():
            registry = str(model_dir)
        else:
            print("=== Caching semantic conventions registry ===")
            semconv_cache.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "-q",
                 "https://github.com/open-telemetry/semantic-conventions.git",
                 str(semconv_cache)],
                check=True,
            )
            registry = str(model_dir)

    # ── Mock server ─────────────────────────────────────────────────

    mock_url = f"http://127.0.0.1:{mock_port}"
    mock_proc: subprocess.Popen | None = None

    try:
        if not is_healthy(f"http://127.0.0.1:{mock_port}/health"):
            print(f"=== Starting mock server on port {mock_port} ===")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-e",
                 str(SCRIPT_DIR / "tests" / "mock-server")],
                capture_output=True,
            )
            mock_proc = subprocess.Popen(
                [sys.executable, str(SCRIPT_DIR / "tests" / "mock-server" / "mock_server" / "server.py")],
            )
            wait_for_health(f"http://127.0.0.1:{mock_port}/health", 30, "Mock server", mock_proc)
        else:
            print(f"Mock server already running on port {mock_port}")

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
        test_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Starting weaver live-check for: {test_name} (ports {weaver_port}/{admin_port}) ===")
        weaver_cmd = ["weaver", "registry", "live-check"]
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

        # ── Run the test ────────────────────────────────────────────

        test_env = {
            **os.environ,
            "MOCK_LLM_URL": mock_url,
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{weaver_port}",
        }

        print(f"=== Running test: {test_name} ===")
        if not run_test_cmd(test_name, test_env):
            print(f"ERROR: Could not find test '{test_name}'", file=sys.stderr)
            print("Available tests:", file=sys.stderr)
            for t in list_available_tests():
                print(f"  {t}", file=sys.stderr)
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

        # ── Update the per-test data file ───────────────────────────

        print("=== Updating test data file ===")
        result = generate_single_test_data(test_name)
        if result is not None:
            path, data = result
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"Updated {path}")
        else:
            print(f"No relevant data for test: {test_name}", file=sys.stderr)

    finally:
        if mock_proc and mock_proc.poll() is None:
            print(f"Stopping mock server (PID {mock_proc.pid})...")
            mock_proc.terminate()
            mock_proc.wait()


if __name__ == "__main__":
    main()
