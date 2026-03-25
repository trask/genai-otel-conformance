"""Test result parsing and span classification."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.metadata import LANGUAGE_DISPLAY_NAMES
from genai_otel_conformance.locations import TestLocation


@dataclass
class SpanClassification:
    detected_types: set[str] = field(default_factory=set)
    per_type_attrs: dict[str, set[str]] = field(default_factory=dict)
    per_type_any_attrs: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class DetectedSignals:
    events: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, int] = field(default_factory=dict)


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
    spans: SpanClassification = field(default_factory=SpanClassification)
    detected: DetectedSignals = field(default_factory=DetectedSignals)

    @property
    def has_detail_content(self) -> bool:
        """Return whether this result contains any renderable detail content."""
        return (
            self.statistics is not None
            or self.has_data
            or bool(self.violation_messages)
            or bool(self.seen_attrs)
            or bool(self.seen_non_registry_attrs)
            or bool(self.seen_events)
            or bool(self.spans.detected_types)
            or bool(self.detected.events)
            or bool(self.detected.metrics)
        )


def split_test_name(name: str) -> tuple[str, str, str]:
    """Parse a test name into language/library/ecosystem slugs."""
    try:
        location = TestLocation.from_test_name(name)
    except ValueError as exc:
        raise ValueError(f"Invalid test name: {name}") from exc

    if location.lang not in LANGUAGE_DISPLAY_NAMES or not location.library or not location.ecosystem:
        raise ValueError(f"Invalid test name: {name}")

    return location.lang, location.library, location.ecosystem


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


def _has_any_attr(span_attrs: dict[str, object], *names: str) -> bool:
    return any(span_attrs.get(name) for name in names)


def _has_all_attrs(span_attrs: dict[str, object], *names: str) -> bool:
    return all(span_attrs.get(name) is not None for name in names)


def _classify_span(span_name: str, span_attrs: dict[str, object]) -> set[str]:
    """Classify a span into span types using heuristics on individual span data."""
    name_lower = span_name.lower()
    op_name = str(span_attrs.get("gen_ai.operation.name", "")).lower()
    oi_kind = str(span_attrs.get("openinference.span.kind", "")).upper()
    llm_type = str(span_attrs.get("llm.request.type", "")).lower()

    matched_types: set[str] = set()

    if (
        "embed" in name_lower
        or _has_any_attr(span_attrs, "embedding.model_name")
        or oi_kind == "EMBEDDING"
        or llm_type in ("embedding", "embeddings")
        or op_name in ("embedding", "embeddings")
    ):
        matched_types.add("embeddings")

    if (
        op_name == "chat"
        or oi_kind == "LLM"
        or llm_type in ("chat", "completion")
        or op_name == "generate_content"
        or _has_all_attrs(span_attrs, "gen_ai.usage.output_tokens", "gen_ai.response.finish_reasons")
        or _has_all_attrs(span_attrs, "llm.response.model", "llm.usage.completion_tokens")
    ):
        matched_types.add("inference")

    if op_name == "create_agent":
        matched_types.add("create_agent")

    if (
        oi_kind == "AGENT"
        or op_name == "invoke_agent"
        or (
            _has_any_attr(span_attrs, "gen_ai.agent.name", "gen_ai.agent.id")
            and op_name != "create_agent"
        )
        or _has_any_attr(span_attrs, "crewai.agent.id", "crewai.agent.role")
        or (
            str(span_attrs.get("rpc.service", "")).lower() == "bedrockagentruntime"
            and str(span_attrs.get("rpc.method", "")).lower() == "invokeagent"
        )
        or ("agentsclient" in name_lower and ("run" in name_lower or "process" in name_lower))
        or ("threads" in name_lower and "run" in name_lower and "thread.run" not in name_lower)
    ):
        matched_types.add("invoke_agent")

    if (
        op_name == "execute_tool"
        or oi_kind == "TOOL"
        or _has_any_attr(span_attrs, "gen_ai.tool.name", "gen_ai.tool.call.id")
    ):
        matched_types.add("execute_tool")

    if (
        op_name == "invoke_workflow"
        or _has_any_attr(span_attrs, "traceloop.workflow.name")
        or name_lower == "crewai.workflow"
        or _has_any_attr(span_attrs, "crewai.crew.id")
    ):
        matched_types.add("invoke_workflow")

    if (
        op_name == "retrieval"
        or oi_kind == "RETRIEVER"
        or _has_any_attr(span_attrs, "gen_ai.data_source.id")
    ):
        matched_types.add("retrieval")

    return matched_types


def _span_attributes(span: dict[str, object]) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for attr in span.get("attributes", []):
        attrs[attr.get("name", "")] = attr.get("value")
    return attrs


def _summarize_samples(
    all_objects: list[dict],
) -> tuple[SpanClassification, DetectedSignals]:
    """Scan sample payloads once and collect detected spans, events, and metrics."""
    spans = SpanClassification()
    signals = DetectedSignals()
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            span = sample.get("span")
            if span:
                attrs = _span_attributes(span)
                classified = _classify_span(span.get("name", ""), attrs)
                spans.detected_types.update(classified)
                attr_names = set(attrs.keys())
                for span_type in classified:
                    if span_type not in spans.per_type_attrs:
                        spans.per_type_attrs[span_type] = set(attr_names)
                    else:
                        spans.per_type_attrs[span_type].intersection_update(attr_names)
                    spans.per_type_any_attrs.setdefault(span_type, set()).update(attr_names)

            log = sample.get("log")
            if log:
                event_name = log.get("event_name", "")
                if event_name.startswith("gen_ai."):
                    signals.events[event_name] = signals.events.get(event_name, 0) + 1

            metric = sample.get("metric")
            if metric:
                metric_name = metric.get("name", "")
                if metric_name.startswith("gen_ai."):
                    signals.metrics[metric_name] = signals.metrics.get(metric_name, 0) + 1

    return spans, signals


def _non_zero_counts(statistics: dict | None, key: str) -> dict[str, int]:
    if not statistics:
        return {}
    return {name: count for name, count in statistics.get(key, {}).items() if count > 0}


def _combined_non_zero_counts(statistics: dict | None, *keys: str) -> dict[str, int]:
    combined: dict[str, int] = {}
    for key in keys:
        combined.update(_non_zero_counts(statistics, key))
    return combined


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


def _merge_detected_signal_counts(
    detected_counts: dict[str, int],
    statistics: dict | None,
    statistics_key: str,
) -> None:
    if not statistics:
        return

    for signal_name, count in statistics.get(statistics_key, {}).items():
        if count <= 0 or not signal_name.startswith("gen_ai."):
            continue
        current_count = detected_counts.get(signal_name, 0)
        if count > current_count:
            detected_counts[signal_name] = count


def parse_result_dir(result_dir: Path, test_name: str) -> TestResult | None:
    """Parse a single test's Weaver output directory into a TestResult."""
    if not result_dir.is_dir():
        return None

    all_objects: list[dict] = []
    for json_file in sorted(result_dir.glob("**/*.json")):
        try:
            all_objects.extend(try_parse_json(json_file.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"Warning: Could not parse {json_file}: {exc}", file=sys.stderr)

    statistics = _extract_statistics(all_objects)

    seen_attrs = _non_zero_counts(statistics, "seen_registry_attributes")
    seen_non_registry_attrs = _non_zero_counts(statistics, "seen_non_registry_attributes")
    seen_events = _combined_non_zero_counts(
        statistics,
        "seen_registry_events",
        "seen_non_registry_events",
    )
    seen_metrics = _combined_non_zero_counts(
        statistics,
        "seen_registry_metrics",
        "seen_non_registry_metrics",
    )

    violation_count = 0
    if statistics:
        violation_count = statistics.get("advice_level_counts", {}).get("violation", 0)

    violation_messages = _violation_messages(statistics)

    entity_counts: dict[str, int] = {}
    if statistics:
        entity_counts = statistics.get("total_entities_by_type", {})

    try:
        lang, library, ecosystem = split_test_name(test_name)
    except ValueError:
        print(f"Warning: Could not parse test name: {test_name}", file=sys.stderr)
        return None
    language = LANGUAGE_DISPLAY_NAMES[lang]

    has_data = False
    if statistics and statistics.get("total_entities", 0) > 0:
        has_data = True
    span_classification, detected = _summarize_samples(all_objects)

    _merge_detected_signal_counts(
        detected.events,
        statistics,
        "seen_non_registry_events",
    )
    _merge_detected_signal_counts(
        detected.metrics,
        statistics,
        "seen_non_registry_metrics",
    )

    return TestResult(
        language=language,
        library=library,
        ecosystem=ecosystem,
        statistics=statistics,
        violation_count=violation_count,
        violation_messages=violation_messages,
        entity_counts=entity_counts,
        seen_attrs=seen_attrs,
        seen_non_registry_attrs=seen_non_registry_attrs,
        seen_events=seen_events,
        seen_metrics=seen_metrics,
        has_data=has_data,
        spans=span_classification,
        detected=detected,
    )