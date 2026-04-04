"""Test data generation and loading from Weaver results."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    SPAN_TYPE_SPECS,
)
from genai_otel_conformance.statuses import (
    build_present_signal_names,
    build_span_type_present_names,
    build_statuses_from_present_names,
    span_type_heatmap_columns,
)


class GeneratedTestData(NamedTuple):
    path: Path
    data: dict[str, object]
    has_relevant_data: bool


@dataclass(frozen=True)
class TestDataEntry:
    test_name: str
    lang: str
    library: str
    ecosystem: str
    library_display: str
    language_display: str
    ecosystem_display: str
    events: dict[str, str]
    metrics: dict[str, str]
    spans: dict[str, dict[str, str]]

    @property
    def label(self) -> str:
        return f"{self.library_display} ({self.language_display}) — {self.ecosystem_display}"

    @classmethod
    def from_result(cls, result: TestResult) -> TestDataEntry:
        language_display = result.language
        return cls(
            test_name=make_anchor_id(language_display, result.library, result.ecosystem),
            lang=LANGUAGE_SLUGS.get(language_display, language_display.lower()),
            library=result.library,
            ecosystem=result.ecosystem,
            library_display=library_display_name(result.library),
            language_display=language_display,
            ecosystem_display=ECOSYSTEM_DISPLAY.get(result.ecosystem, result.ecosystem),
            events={},
            metrics={},
            spans={},
        )


def _normalize_generated_test_payload(data: dict[str, object]) -> dict[str, object]:
    """Drop empty top-level objects and sort span attribute names alphabetically."""
    normalized: dict[str, object] = {}
    spans = data.get("spans")
    if spans:
        cleaned = {
            span_type: sorted(attrs)
            for span_type, attrs in spans.items()
            if attrs
        }
        if cleaned:
            normalized["spans"] = cleaned
    for key in ("metrics", "events"):
        value = data.get(key)
        if value:
            normalized[key] = {
                name: []
                for name in _present_signal_names(value)
            }
    return normalized


def _build_single_test_data(test_name: str, result: TestResult) -> GeneratedTestData:
    """Build committed dashboard data from a parsed Weaver result."""
    event_names = build_present_signal_names(
        GENAI_EVENT_TYPES,
        result.observed.events,
        result.detected.events,
    )
    metric_names = build_present_signal_names(
        GENAI_METRIC_TYPES,
        result.observed.metrics,
        result.detected.metrics,
    )
    has_genai_signals = bool(event_names) or bool(metric_names)
    spans = build_span_type_present_names(result)
    path = TestLocation.from_test_name(test_name).data_file(TESTS_DIR)

    data: dict[str, object] = {
        "events": event_names,
        "metrics": metric_names,
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
    field_name: str,
    test_name: str,
) -> dict[str, str]:
    return build_statuses_from_present_names(
        signal_names,
        _signal_names_from_committed_data(value, field_name, test_name),
    )


def _present_signal_names(value: dict | list | None) -> list[str]:
    if not isinstance(value, (dict, list)):
        return []
    return sorted(name for name in value if isinstance(name, str))


def _signal_names_from_committed_data(
    value: object,
    field_name: str,
    test_name: str,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise ValueError(
            f"Invalid {field_name} data for {test_name}: expected an object mapping signal names to lists"
        )

    for name, payload in value.items():
        if not isinstance(name, str) or not isinstance(payload, list):
            raise ValueError(
                f"Invalid {field_name} data for {test_name}: expected an object mapping signal names to lists"
            )

    return _present_signal_names(value)


def _normalize_span_type_data(value: dict | None) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for span_type_key, spec in SPAN_TYPE_SPECS.items():
        if span_type_key not in value:
            continue

        expected_names = [column.header_text for column in span_type_heatmap_columns(spec)]
        raw = value[span_type_key]
        present_names = [name for name in raw if isinstance(name, str)] if isinstance(raw, (dict, list)) else []
        normalized[span_type_key] = build_statuses_from_present_names(expected_names, present_names)

    return normalized


def _display_value(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _normalize_test_data_entry(entry: dict[str, object], location: TestLocation) -> TestDataEntry:
    language_display = _display_value(
        entry.get("language"),
        LANGUAGE_DISPLAY_NAMES.get(location.lang, location.lang),
    )
    test_name = make_anchor_id(language_display, location.library, location.ecosystem)
    return TestDataEntry(
        test_name=test_name,
        lang=location.lang,
        library=location.library,
        ecosystem=location.ecosystem,
        library_display=_display_value(entry.get("library"), library_display_name(location.library)),
        language_display=language_display,
        ecosystem_display=_display_value(
            entry.get("ecosystem"),
            ECOSYSTEM_DISPLAY.get(location.ecosystem, location.ecosystem),
        ),
        events=_normalize_signal_data(entry.get("events"), GENAI_EVENT_TYPES, "events", test_name),
        metrics=_normalize_signal_data(entry.get("metrics"), GENAI_METRIC_TYPES, "metrics", test_name),
        spans=_normalize_span_type_data(entry.get("spans")),
    )


def load_test_data_files() -> list[TestDataEntry]:
    """Discover and load committed data-*.json files from tests/.

    Only files in the canonical tests/<lang>/<lib>/data-<ecosystem>.json layout
    are considered. This avoids pulling in copied workspace artifacts such as
    tests/js/node_modules/*/data-*.json.

    Returns normalized typed entries for dashboard generation.
    """
    entries: list[TestDataEntry] = []
    if not TESTS_DIR.is_dir():
        return entries
    for data_file in sorted(TESTS_DIR.glob("*/*/data-*.json")):
        data = json.loads(data_file.read_text(encoding="utf-8"))
        location = TestLocation.from_data_file(data_file, TESTS_DIR)
        entries.append(_normalize_test_data_entry(data, location))
    return entries
