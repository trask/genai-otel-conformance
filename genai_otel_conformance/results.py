"""Test result parsing."""

from __future__ import annotations

from collections.abc import Iterable
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


def merge_signal_counts(
    statistics_counts: dict[str, int],
    detected_counts: dict[str, int],
) -> dict[str, int]:
    """Merge statistic-derived and sample-derived signal counts."""
    merged = dict(statistics_counts)
    for name, count in detected_counts.items():
        merged[name] = max(merged.get(name, 0), count)
    return merged


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


def _load_result_objects(result_dir: Path) -> list[dict]:
    """Load and parse all JSON result objects from a Weaver result directory."""
    all_objects: list[dict] = []
    for json_file in sorted(result_dir.glob("**/*.json")):
        all_objects.extend(try_parse_json(json_file.read_text(encoding="utf-8")))
    return all_objects


_PRESENCE_BLOCKING_ADVICE_IDS = {"type_mismatch"}


def _attribute_blocks_presence(attribute: dict[str, object]) -> bool:
    live_check_result = attribute.get("live_check_result")
    if not isinstance(live_check_result, dict):
        return False

    for advice in live_check_result.get("all_advice", []):
        if not isinstance(advice, dict):
            continue
        if advice.get("id") == "not_stable":
            continue
        if advice.get("id") in _PRESENCE_BLOCKING_ADVICE_IDS:
            return True
        if advice.get("level") == "violation":
            return True

    return False


def _attribute_counts_as_present(attribute: dict[str, object]) -> bool:
    return not _attribute_blocks_presence(attribute)


def _iter_attribute_records(node: object) -> Iterable[dict[str, object]]:
    if isinstance(node, dict):
        attrs = node.get("attributes")
        if isinstance(attrs, list):
            for attr in attrs:
                if isinstance(attr, dict):
                    yield attr
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from _iter_attribute_records(value)
        return

    if isinstance(node, list):
        for value in node:
            if isinstance(value, (dict, list)):
                yield from _iter_attribute_records(value)


def _observed_registry_attribute_counts_from_samples(all_objects: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            for attr in _iter_attribute_records(sample):
                name = attr.get("name")
                if not isinstance(name, str) or not name:
                    continue
                if not _attribute_counts_as_present(attr):
                    continue
                counts[name] = counts.get(name, 0) + 1
    return counts


def _observed_telemetry_from_statistics(
    statistics: dict | None,
    all_objects: list[dict],
) -> ObservedTelemetry:
    """Build observed telemetry counts from Weaver summary statistics."""
    seen_events = _non_zero_counts(statistics, "seen_registry_events")
    seen_events.update(_non_zero_counts(statistics, "seen_non_registry_events"))

    seen_metrics = _non_zero_counts(statistics, "seen_registry_metrics")
    seen_metrics.update(_non_zero_counts(statistics, "seen_non_registry_metrics"))

    seen_registry_attrs = _non_zero_counts(statistics, "seen_registry_attributes")
    sample_registry_attrs = _observed_registry_attribute_counts_from_samples(all_objects)
    if sample_registry_attrs:
        if seen_registry_attrs:
            seen_registry_attrs = {
                name: sample_registry_attrs.get(name, 0)
                for name in seen_registry_attrs
                if sample_registry_attrs.get(name, 0) > 0
            }
        else:
            seen_registry_attrs = {
                name: count
                for name, count in sample_registry_attrs.items()
                if count > 0
            }

    entity_counts: dict[str, int] = {}
    has_data = False
    if statistics:
        entity_counts = statistics.get("total_entities_by_type", {})
        has_data = statistics.get("total_entities", 0) > 0

    return ObservedTelemetry(
        attrs=seen_registry_attrs,
        non_registry_attrs=_non_zero_counts(statistics, "seen_non_registry_attributes"),
        events=seen_events,
        metrics=seen_metrics,
        entity_counts=entity_counts,
        has_data=has_data,
    )


def _detected_signals_from_samples(
    all_objects: list[dict],
    statistics: dict | None,
) -> tuple[SpanClassification, DetectedSignals]:
    """Classify spans and supplement detected signal counts from statistics."""
    span_classification, detected = summarize_samples(
        all_objects,
        include_attr=_attribute_counts_as_present,
    )
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
    return span_classification, detected


def _build_test_result(
    location: TestLocation,
    statistics: dict | None,
    observed: ObservedTelemetry,
    spans: SpanClassification,
    detected: DetectedSignals,
) -> TestResult:
    """Assemble the final parsed test result model."""
    _validate_test_lang(location)
    language = LANGUAGE_DISPLAY_NAMES[location.lang]
    violation_count = 0
    if statistics:
        violation_count = statistics.get("advice_level_counts", {}).get("violation", 0)

    return TestResult(
        language=language,
        library=location.library,
        ecosystem=location.ecosystem,
        statistics=statistics,
        violation_count=violation_count,
        violation_messages=_violation_messages(statistics),
        observed=observed,
        spans=spans,
        detected=detected,
    )


def parse_result_dir(result_dir: Path, test_name: str) -> TestResult | None:
    """Parse a single test's Weaver output directory into a TestResult."""
    if not result_dir.is_dir():
        return None

    all_objects = _load_result_objects(result_dir)
    statistics = _extract_statistics(all_objects)
    location = TestLocation.from_test_name(test_name)
    observed = _observed_telemetry_from_statistics(statistics, all_objects)
    span_classification, detected = _detected_signals_from_samples(all_objects, statistics)
    return _build_test_result(location, statistics, observed, span_classification, detected)


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