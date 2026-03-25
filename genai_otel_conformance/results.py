"""Test result parsing."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.classification import (
    DetectedSignals,
    SpanClassification,
    summarize_samples,
)
from genai_otel_conformance.metadata import LANGUAGE_DISPLAY_NAMES
from genai_otel_conformance.locations import TestLocation


def _validate_test_lang(location: TestLocation) -> None:
    """Raise ValueError if the test location uses an unknown language."""
    if location.lang not in LANGUAGE_DISPLAY_NAMES:
        raise ValueError(f"Invalid test name: {location.test_name}")


@dataclass
class ObservedTelemetry:
    attrs: dict[str, int] = field(default_factory=dict)
    non_registry_attrs: dict[str, int] = field(default_factory=dict)
    events: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, int] = field(default_factory=dict)
    entity_counts: dict[str, int] = field(default_factory=dict)
    has_data: bool = False


@dataclass
class TestResult:
    language: str
    library: str
    ecosystem: str
    statistics: dict | None
    violation_count: int
    violation_messages: list[str]
    observed: ObservedTelemetry = field(default_factory=ObservedTelemetry)
    spans: SpanClassification = field(default_factory=SpanClassification)
    detected: DetectedSignals = field(default_factory=DetectedSignals)

    @property
    def has_detail_content(self) -> bool:
        """Return whether this result contains any renderable detail content."""
        return (
            self.statistics is not None
            or self.observed.has_data
            or bool(self.violation_messages)
            or bool(self.observed.attrs)
            or bool(self.observed.non_registry_attrs)
            or bool(self.observed.events)
            or bool(self.spans.detected_types)
            or bool(self.detected.events)
            or bool(self.detected.metrics)
        )


def try_parse_json(content: str) -> list[dict]:
    """Parse JSON content, handling a single object, array, or JSONL."""
    objects: list[dict] = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        objects.extend(data)
        return objects
    if isinstance(data, dict):
        objects.append(data)
        return objects

    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return objects


def _non_zero_counts(statistics: dict | None, key: str) -> dict[str, int]:
    if not statistics:
        return {}
    return {name: count for name, count in statistics.get(key, {}).items() if count > 0}


def _extract_statistics(all_objects: list[dict]) -> dict | None:
    statistics = None
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        if "statistics" in obj and isinstance(obj["statistics"], dict):
            statistics = obj["statistics"]
            continue
        if "registry_coverage" in obj or "advice_level_counts" in obj:
            statistics = obj
    return statistics


def _violation_messages(statistics: dict | None) -> list[str]:
    if not statistics:
        return []

    messages: set[str] = set()
    for message in statistics.get("advice_message_counts", {}):
        if "not stable" in message.lower():
            continue
        messages.add(message)
    return sorted(messages)


def _supplement_detected_from_statistics(
    detected_counts: dict[str, int],
    statistics: dict | None,
    statistics_key: str,
) -> dict[str, int]:
    """Supplement sample-derived signal counts with gen_ai.* counts from statistics."""
    merged = dict(detected_counts)
    if not statistics:
        return merged

    for signal_name, count in statistics.get(statistics_key, {}).items():
        if count <= 0 or not signal_name.startswith("gen_ai."):
            continue
        if count > merged.get(signal_name, 0):
            merged[signal_name] = count
    return merged


def parse_result_dir(result_dir: Path, test_name: str) -> TestResult | None:
    """Parse a single test's Weaver output directory into a TestResult."""
    if not result_dir.is_dir():
        return None

    all_objects: list[dict] = []
    for json_file in sorted(result_dir.glob("**/*.json")):
        all_objects.extend(try_parse_json(json_file.read_text(encoding="utf-8")))

    statistics = _extract_statistics(all_objects)

    seen_attrs = _non_zero_counts(statistics, "seen_registry_attributes")
    seen_non_registry_attrs = _non_zero_counts(statistics, "seen_non_registry_attributes")
    seen_events = _non_zero_counts(statistics, "seen_registry_events")
    seen_events.update(_non_zero_counts(statistics, "seen_non_registry_events"))
    seen_metrics = _non_zero_counts(statistics, "seen_registry_metrics")
    seen_metrics.update(_non_zero_counts(statistics, "seen_non_registry_metrics"))

    violation_count = 0
    if statistics:
        violation_count = statistics.get("advice_level_counts", {}).get("violation", 0)

    violation_messages = _violation_messages(statistics)

    entity_counts: dict[str, int] = {}
    if statistics:
        entity_counts = statistics.get("total_entities_by_type", {})

    location = TestLocation.from_test_name(test_name)
    _validate_test_lang(location)
    language = LANGUAGE_DISPLAY_NAMES[location.lang]

    has_data = False
    if statistics and statistics.get("total_entities", 0) > 0:
        has_data = True
    span_classification, detected = summarize_samples(all_objects)

    detected.events = _supplement_detected_from_statistics(
        detected.events,
        statistics,
        "seen_non_registry_events",
    )
    detected.metrics = _supplement_detected_from_statistics(
        detected.metrics,
        statistics,
        "seen_non_registry_metrics",
    )

    return TestResult(
        language=language,
        library=location.library,
        ecosystem=location.ecosystem,
        statistics=statistics,
        violation_count=violation_count,
        violation_messages=violation_messages,
        observed=ObservedTelemetry(
            attrs=seen_attrs,
            non_registry_attrs=seen_non_registry_attrs,
            events=seen_events,
            metrics=seen_metrics,
            entity_counts=entity_counts,
            has_data=has_data,
        ),
        spans=span_classification,
        detected=detected,
    )


def parse_all_results() -> dict[str, TestResult]:
    """Parse all Weaver output directories under tests/.

    Layout: tests/<lang>/<lib>/results/<eco>/
    """
    results: dict[str, TestResult] = {}

    if not TESTS_DIR.exists():
        print(f"Tests directory not found: {TESTS_DIR}", file=sys.stderr)
        return results

    result_dirs = [p for p in TESTS_DIR.glob("*/*/results/*") if p.is_dir()]

    for result_dir in sorted(result_dirs):
        location = TestLocation.from_results_dir(result_dir, TESTS_DIR)
        result = parse_result_dir(result_dir, location.test_name)
        if result is None or not result.has_detail_content:
            continue
        results[location.test_name] = result

    return results