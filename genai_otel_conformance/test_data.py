"""Test data generation from Weaver results."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.locations import TestLocation
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
)


class GeneratedTestData(NamedTuple):
    path: Path
    data: dict[str, object]
    has_relevant_data: bool


def _normalize_generated_test_payload(data: dict[str, object]) -> dict[str, object]:
    """Drop empty top-level objects and sort span attribute names alphabetically."""
    normalized: dict[str, object] = {}
    if spans := data.get("spans"):
        cleaned = {
            span_type: sorted(attrs)
            for span_type, attrs in spans.items()
            if attrs
        }
        if cleaned:
            normalized["spans"] = cleaned
    for key in ("events", "metrics"):
        if value := data.get(key):
            normalized[key] = dict(sorted(value.items()))
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
