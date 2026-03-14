#!/usr/bin/env python3
"""Generate a static HTML conformance dashboard from Weaver live-check results.

Usage:
    python generate_dashboard.py [--results-dir RESULTS] [--output-dir OUTPUT]

Reads Weaver JSON output from results/<test-name>/ directories and produces
a static HTML dashboard at <output-dir>/index.html.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import jinja2

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"

# ── Test name parsing ────────────────────────────────────────────────

ECOSYSTEM_SUFFIXES = ["otelcontrib", "openllmetry", "openinference", "native"]

# New-style test name language prefixes (e.g. "python-cohere-openllmetry").
LANG_PREFIXES = {
    "python-": "Python",
    "js-": "JS",
    "java-": "Java",
    "dotnet-": "C#",
}

ECOSYSTEM_DISPLAY = {
    "otelcontrib": "OTel Contrib",
    "openllmetry": "OpenLLMetry",
    "openinference": "OpenInference",
    "native": "Native",
}

# Maps (ecosystem, language) to the instrumentation repository slug.
# For "native" the repo depends on the library, handled via NATIVE_REPOS below.
ECOSYSTEM_REPOS = {
    ("otelcontrib", "Python"): "open-telemetry/opentelemetry-python-contrib",
    ("otelcontrib", "Java"): "open-telemetry/opentelemetry-java-instrumentation",
    ("otelcontrib", "JS"): "open-telemetry/opentelemetry-js-contrib",
    ("otelcontrib", "C#"): "open-telemetry/opentelemetry-dotnet-contrib",
    ("openllmetry", "Python"): "traceloop/openllmetry",
    ("openllmetry", "JS"): "traceloop/openllmetry",
    ("openinference", "Python"): "Arize-ai/openinference",
    ("openinference", "JS"): "Arize-ai/openinference",
}

# For "native" ecosystem, the repo depends on the library.
NATIVE_REPOS = {
    "autogen": "microsoft/autogen",
    "extensions-ai": "dotnet/extensions",
    "google-adk": "google/adk-python",
    "spring-ai": "spring-projects/spring-ai",
    "spring-ai-openinference": "spring-projects/spring-ai",
    "litellm": "BerriAI/litellm",
    "dspy": "stanfordnlp/dspy",
    "vercel-ai": "vercel/ai",
    "pydantic-ai": "pydantic/pydantic-ai",
    "semantic-kernel": "microsoft/semantic-kernel",
    "promptflow": "microsoft/promptflow",
    "openai-agents": "openai/openai-agents-python",
}

# ── Library display names (discovered from tests/<language>/<library>/metadata.json) ─

TESTS_DIR = SCRIPT_DIR / "tests"

# Language directory names → display language names.
_LANG_DIRS = {"python": "Python", "java": "Java", "js": "JS", "dotnet": "C#"}


def _discover_library_display_names() -> dict[str, str]:
    """Scan tests/<language>/<library>/metadata.json for display_name entries.

    Returns a dict mapping library directory slug → display name.
    Libraries that appear under multiple languages only need one metadata.json
    with a display_name (the first one found wins, they should all agree).
    Falls back to the slug itself when no metadata is found.
    """
    names: dict[str, str] = {}
    if not TESTS_DIR.is_dir():
        return names
    for lang_dir in sorted(TESTS_DIR.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name not in _LANG_DIRS:
            continue
        for lib_dir in sorted(lang_dir.iterdir()):
            if not lib_dir.is_dir():
                continue
            slug = lib_dir.name
            if slug in names:
                continue
            meta = lib_dir / "metadata.json"
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                if "display_name" in data:
                    names[slug] = data["display_name"]
            except (OSError, json.JSONDecodeError):
                pass
    return names


_LIBRARY_DISPLAY_NAMES: dict[str, str] = _discover_library_display_names()


def _library_display_name(slug: str) -> str:
    """Return the human-readable display name for a library slug."""
    return _LIBRARY_DISPLAY_NAMES.get(slug, slug)

# Deprecated attribute alternatives: maps new name -> old name.
# Shown as an extra column in the heatmap, highlighted yellow when present.
DEPRECATED_ALTERNATIVES = {
    "gen_ai.provider.name": "gen_ai.system",
}

# ── Span type specifications (from gen-ai-spans.md and gen-ai-agent-spans.md) ─

# Expected attributes per span type, grouped by requirement level.
# Source: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
# Source: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-agent-spans.md

# Shared attribute lists to avoid repetition across span type specs.
_COMMON_REQUIRED = ["gen_ai.operation.name"]
_PROVIDER_REQUIRED = ["gen_ai.provider.name"]
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
    "invoke_agent": {
        "label": "Invoke Agent",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.agent.id", "gen_ai.agent.name",
        },
        # Non-standard attributes that indicate agent workflow but don't follow
        # the spec.  Tests matching only these show up in the heatmap to
        # highlight the conformance gap.
        "non_standard_discriminator_attrs": {
            "crewai.agent.id", "crewai.agent.role",
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


@dataclass
class HeatmapRow:
    test_name: str
    lib_display: str
    eco_display: str
    present_attrs: set[str]
    library: str
    language: str
    repo: str
    instrumentation_version: str


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
        lib_norm = library.lower().replace("-", "")
        best = None
        for pkg, ver in versions.items():
            pkg_lower = pkg.lower().replace("-", "").replace("_", "")
            if any(pkg_lower.startswith(ip) for ip in _INFRA_PREFIXES):
                continue
            if lib_norm in pkg_lower or pkg_lower in lib_norm:
                if best is None or len(pkg) > len(best[0]):
                    best = (pkg, ver)
        if best:
            return f"{best[0]} {best[1]}"

    return ""


def parse_results(results_dir):
    """Parse all Weaver output directories under results_dir.

    Weaver live-check output format:
      {
        "samples": [ { "resource": { "attributes": [...], "live_check_result": {...} } }, ... ],
        "statistics": {
          "total_entities": N,
          "total_entities_by_type": {"span": N, "log": N, ...},
          "advice_level_counts": {"violation": N, "improvement": N},
          "advice_type_counts": {"deprecated": N, "not_stable": N},
          "advice_message_counts": {"msg": count, ...},
          "seen_registry_attributes": {"attr.name": count, ...},
          "seen_non_registry_attributes": {"attr.name": count, ...},
          "seen_registry_events": {"event.name": count, ...},
          "seen_non_registry_events": {"event.name": count, ...},
          "registry_coverage": float,
          ...
        }
      }
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return results

    for test_dir in sorted(results_path.iterdir()):
        if not test_dir.is_dir():
            continue

        test_name = test_dir.name
        all_objects = []

        for json_file in sorted(test_dir.glob("**/*.json")):
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
        versions_file = test_dir / "versions.json"
        if versions_file.exists():
            try:
                versions = json.loads(versions_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        results[test_name] = TestResult(
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
        )

    return results


# ── HTML generation ──────────────────────────────────────────────────

SPAN_TYPE_ORDER = ["invoke_agent", "inference", "embeddings", "retrieval", "execute_tool"]


def _build_heatmap_rows(results: dict[str, TestResult]) -> list[HeatmapRow]:
    """Build one HeatmapRow per test that has gen_ai.* attribute data."""
    rows = []
    for test_name, r in results.items():
        if not r.has_data:
            continue
        all_present = set(r.seen_attrs) | set(r.seen_non_registry_attrs)
        if not any(a.startswith("gen_ai.") for a in all_present):
            continue

        lib_display = _library_display_name(r.library)
        eco_display = ECOSYSTEM_DISPLAY.get(r.ecosystem, r.ecosystem)
        if r.ecosystem == "native":
            repo = NATIVE_REPOS.get(r.library, r.ecosystem)
        else:
            repo = ECOSYSTEM_REPOS.get(
                (r.ecosystem, r.language),
                ECOSYSTEM_DISPLAY.get(r.ecosystem, r.ecosystem),
            )

        rows.append(HeatmapRow(
            test_name=test_name,
            lib_display=lib_display,
            eco_display=eco_display,
            present_attrs=all_present,
            library=r.library,
            language=r.language,
            repo=repo,
            instrumentation_version=_extract_instrumentation_version(
                r.ecosystem, r.library, r.versions,
            ),
        ))

    rows.sort(key=lambda x: (x.language.lower(), x.lib_display.lower(), x.eco_display.lower()))
    return rows


def _prepare_heatmaps(heatmap_rows: list[HeatmapRow]) -> list[dict]:
    """Prepare per-span-type heatmap data for the template."""
    heatmaps = []

    for st_key in SPAN_TYPE_ORDER:
        spec = SPAN_TYPE_SPECS[st_key]

        # Build column definitions
        columns = []
        col_defs: list[tuple[str, bool]] = []  # (attr_name, is_group_start)
        for level in ("required", "conditionally_required", "recommended"):
            attrs = spec.get(level, [])
            for i, attr in enumerate(attrs):
                is_group_start = i == 0
                if attr in DEPRECATED_ALTERNATIVES:
                    header_text = f"{attr} / {DEPRECATED_ALTERNATIVES[attr]}"
                else:
                    header_text = attr
                columns.append({"header_text": header_text, "is_group_start": is_group_start})
                col_defs.append((attr, is_group_start))

        if not columns:
            continue

        # All spec attrs for filtering
        all_spec_attrs: set[str] = set()
        for level in ("required", "conditionally_required", "recommended"):
            all_spec_attrs.update(spec.get(level, []))

        # Filter rows by discriminator attributes
        discriminators = spec.get("discriminator_attrs", set())
        non_std_discriminators = spec.get("non_standard_discriminator_attrs", set())
        if discriminators:
            relevant = [r for r in heatmap_rows
                        if r.present_attrs & (discriminators | non_std_discriminators)]
        else:
            relevant = [r for r in heatmap_rows if r.present_attrs & all_spec_attrs]

        if not relevant:
            continue

        # Build row data with pre-computed cells
        rows = []
        for row in relevant:
            cells = []
            for attr, is_group_start in col_defs:
                present = attr in row.present_attrs
                if attr in DEPRECATED_ALTERNATIVES:
                    dep_name = DEPRECATED_ALTERNATIVES[attr]
                    dep_present = dep_name in row.present_attrs
                    if present:
                        cls, symbol = "present", "\u2713"
                    elif dep_present:
                        cls, symbol = "deprecated", "\u2713"
                    else:
                        cls, symbol = "absent", ""
                else:
                    cls = "present" if present else "absent"
                    symbol = "\u2713" if present else ""
                group_cls = " group-start" if is_group_start else ""
                cells.append({"cls": f"{cls}{group_cls}", "symbol": symbol})

            # For the heatmap, show just the version number (the full
            # package name is redundant next to the ecosystem label).
            short_version = ""
            if row.instrumentation_version:
                parts = row.instrumentation_version.rsplit(" ", 1)
                short_version = parts[-1] if len(parts) == 2 else row.instrumentation_version

            rows.append({
                "test_name": row.test_name,
                "lib_display": row.lib_display,
                "language": row.language,
                "eco_display": row.eco_display,
                "instrumentation_version": short_version,
                "cells": cells,
            })

        # Compute rowspan values for Language → Library hierarchy.
        # lang_rowspan > 0 means "emit a language cell with this rowspan";
        # lib_rowspan > 0 means "emit a library cell with this rowspan";
        # 0 means "skip the cell (covered by a previous rowspan)".
        for r in rows:
            r["lang_rowspan"] = 0
            r["lib_rowspan"] = 0

        if rows:
            i = 0
            while i < len(rows):
                # Find the extent of this language group.
                lang = rows[i]["language"]
                lang_start = i
                while i < len(rows) and rows[i]["language"] == lang:
                    i += 1
                lang_end = i
                rows[lang_start]["lang_rowspan"] = lang_end - lang_start

                # Within the language group, find library sub-groups.
                j = lang_start
                while j < lang_end:
                    lib = rows[j]["lib_display"]
                    lib_start = j
                    while j < lang_end and rows[j]["lib_display"] == lib:
                        j += 1
                    rows[lib_start]["lib_rowspan"] = j - lib_start

        heatmaps.append({"label": spec["label"], "columns": columns, "rows": rows})

    return heatmaps


def _prepare_details(results: dict[str, TestResult]) -> list[dict]:
    """Prepare detailed result data for the template."""
    details = []

    for test_name in sorted(results):
        r = results[test_name]
        lib_display = _library_display_name(r.library)
        eco_display = ECOSYSTEM_DISPLAY.get(r.ecosystem, r.ecosystem)
        label = f"{lib_display} ({r.language}) \u2014 {eco_display}"

        instrumentation_version = _extract_instrumentation_version(
            r.ecosystem, r.library, r.versions,
        )

        if r.ecosystem == "native":
            repo = NATIVE_REPOS.get(r.library, "")
        else:
            repo = ECOSYSTEM_REPOS.get((r.ecosystem, r.language), "")

        detail: dict = {
            "test_name": test_name,
            "label": label,
            "has_data": r.has_data,
            "violation_count": r.violation_count,
            "instrumentation_version": instrumentation_version,
            "repo": repo,
            "entity_summary": "",
            "span_sections": [],
            "non_registry_attrs": [],
            "events": [],
            "violation_messages": [],
        }

        if r.has_data:
            # Entity summary
            entity_parts = []
            for etype in ("span", "log", "resource", "attribute"):
                count = r.entity_counts.get(etype, 0)
                if count > 0:
                    entity_parts.append(f"{count} {etype}{'s' if count != 1 else ''}")
            detail["entity_summary"] = ", ".join(entity_parts)

            # Span-type attribute checklists
            all_present = set(r.seen_attrs) | set(r.seen_non_registry_attrs)

            for st_key in SPAN_TYPE_ORDER:
                spec = SPAN_TYPE_SPECS[st_key]
                all_spec_attrs: set[str] = set()
                for level in ("required", "conditionally_required", "recommended"):
                    all_spec_attrs.update(spec.get(level, []))

                discriminators = spec.get("discriminator_attrs", set())
                non_std_discriminators = spec.get("non_standard_discriminator_attrs", set())
                if discriminators:
                    if not (all_present & (discriminators | non_std_discriminators)):
                        continue
                elif not (all_present & all_spec_attrs):
                    continue

                groups = []
                for level, level_label in [("required", "Required"),
                                           ("conditionally_required", "Conditionally Required"),
                                           ("recommended", "Recommended")]:
                    expected = spec.get(level, [])
                    if not expected:
                        continue
                    attrs = []
                    for attr in expected:
                        if attr in all_present:
                            count = r.seen_attrs.get(attr, r.seen_non_registry_attrs.get(attr, 0))
                            attrs.append({"name": attr, "present": True, "count": count})
                        else:
                            attrs.append({"name": attr, "present": False, "count": 0})
                    groups.append({"label": level_label, "attrs": attrs})

                detail["span_sections"].append({"label": spec["label"], "groups": groups})

            # Non-registry attributes
            if r.seen_non_registry_attrs:
                detail["non_registry_attrs"] = [
                    {"name": a, "count": c}
                    for a, c in sorted(r.seen_non_registry_attrs.items())
                ]

            # Events
            if r.seen_events:
                detail["events"] = [
                    {"name": a, "count": c}
                    for a, c in sorted(r.seen_events.items())
                ]

        # Violation messages
        if r.violation_messages:
            detail["violation_messages"] = r.violation_messages

        details.append(detail)

    return details


def generate_html(results: dict[str, TestResult]) -> str:
    """Generate the full HTML dashboard string using Jinja2."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    heatmap_rows = _build_heatmap_rows(results)
    heatmaps = _prepare_heatmaps(heatmap_rows)
    details = _prepare_details(results)

    css = (TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("dashboard.html")
    return template.render(css=css, now=now, heatmaps=heatmaps, details=details)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate conformance dashboard")
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing Weaver output subdirectories (default: results)",
    )
    parser.add_argument(
        "--output-dir", default="build",
        help="Directory to write index.html (default: build)",
    )
    args = parser.parse_args()

    results = parse_results(args.results_dir)

    if not results:
        print("No results found. Generating empty dashboard.", file=sys.stderr)

    html_content = generate_html(results)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html_content, encoding="utf-8")

    print(f"Dashboard written to {out / 'index.html'}")
    print(f"  Tests: {len(results)}")


if __name__ == "__main__":
    main()
