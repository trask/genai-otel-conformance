"""Test data generation and loading from Weaver results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.locations import TestLocation
from genai_otel_conformance.metadata import (
    ECOSYSTEM_DISPLAY,
    LANGUAGE_DISPLAY_NAMES,
    LANGUAGE_SLUGS,
    library_display_name,
)
from genai_otel_conformance.results import (
    TestResult,
    parse_result_dir,
)
from genai_otel_conformance.specs import (
    GENAI_EVENT_TYPES,
    GENAI_METRIC_TYPES,
    SPAN_TYPE_ORDER,
    SPAN_TYPE_SPECS,
)
from genai_otel_conformance.statuses import (
    build_present_signal_entries,
    build_span_type_present_names,
    build_statuses_from_present_names,
    span_type_heatmap_columns,
)


class GeneratedTestData(NamedTuple):
    path: Path
    data: dict[str, object]
    has_relevant_data: bool


def _normalize_generated_test_payload(data: dict[str, object]) -> dict[str, object]:
    """Drop empty top-level objects and sort span attribute names alphabetically."""
    normalized: dict[str, object] = {}
    for key in ("events", "metrics"):
        if value := data.get(key):
            normalized[key] = dict(sorted(value.items()))
    if spans := data.get("spans"):
        cleaned = {
            span_type: sorted(attrs)
            for span_type, attrs in spans.items()
            if attrs
        }
        if cleaned:
            normalized["spans"] = cleaned
    return normalized


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
    path = TestLocation.from_test_name(test_name).data_file(TESTS_DIR)

    data: dict[str, object] = {
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


def generate_single_test_data(test_name: str) -> GeneratedTestData | None:
    """Generate data for a single test from its results directory.

    Returns generated dashboard data or None if the Weaver output could not be parsed.
    """
    result_dir = TestLocation.from_test_name(test_name).results_dir(TESTS_DIR)
    result = parse_result_dir(result_dir, test_name)
    if result is None:
        return None
    return _build_single_test_data(test_name, result)


# ── Data file loading and normalization ─────────────────────────────


def make_anchor_id(language: str, library: str, ecosystem: str) -> str:
    """Build an anchor ID in library-language-ecosystem order.

    This matches the dashboard's visual hierarchy: group by library,
    then by language, then by ecosystem.

    Args:
        language: Display language name (e.g. "Python", "JS").
        library: Library slug (e.g. "openai").
        ecosystem: Ecosystem slug (e.g. "otelcontrib").
    """
    lang_slug = LANGUAGE_SLUGS.get(language, language.lower())
    return f"{library}-{lang_slug}-{ecosystem}"


def _normalize_signal_data(
    value: object,
    signal_names: list[str],
) -> dict[str, str]:
    present = [name for name in value if isinstance(name, str)] if isinstance(value, (dict, list)) else []
    return build_statuses_from_present_names(signal_names, present)


def _normalize_span_type_data(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for span_type_key, spec in SPAN_TYPE_SPECS.items():
        if span_type_key not in value:
            continue

        expected_names = [column["header_text"] for column in span_type_heatmap_columns(spec)]
        raw = value[span_type_key]
        present_names = [name for name in raw if isinstance(name, str)] if isinstance(raw, (dict, list)) else []
        normalized[span_type_key] = build_statuses_from_present_names(expected_names, present_names)

    return normalized


def _normalize_test_data_entry(entry: dict) -> dict:
    normalized = dict(entry)
    normalized["events"] = _normalize_signal_data(entry.get("events"), GENAI_EVENT_TYPES)
    normalized["metrics"] = _normalize_signal_data(entry.get("metrics"), GENAI_METRIC_TYPES)
    raw_spans = entry.get("spans") or entry.get("span_types")
    normalized["spans"] = _normalize_span_type_data(raw_spans)
    normalized.pop("span_types", None)
    return normalized


def load_test_data_files() -> list[dict]:
    """Discover and load committed data-*.json files from tests/.

    Only files in the canonical tests/<lang>/<lib>/data-<ecosystem>.json layout
    are considered. This avoids pulling in copied workspace artifacts such as
    tests/js/node_modules/*/data-*.json.

    Returns a list of loaded data dicts (each augmented with the file path).
    """
    entries: list[dict] = []
    if not TESTS_DIR.is_dir():
        return entries
    for data_file in sorted(TESTS_DIR.glob("*/*/data-*.json")):
        data = json.loads(data_file.read_text(encoding="utf-8"))
        location = TestLocation.from_data_file(data_file, TESTS_DIR)
        data.setdefault("library", library_display_name(location.library))
        data.setdefault("language", LANGUAGE_DISPLAY_NAMES.get(location.lang, location.lang))
        data.setdefault(
            "ecosystem",
            ECOSYSTEM_DISPLAY.get(location.ecosystem, location.ecosystem),
        )
        data["test_name"] = make_anchor_id(data["language"], location.library, location.ecosystem)
        data["_lang"] = location.lang
        data["_lib"] = location.library
        data["_eco"] = location.ecosystem
        entries.append(_normalize_test_data_entry(data))
    return entries
