"""Build heatmap statuses and attribute lookups from test results."""

from __future__ import annotations

from dataclasses import dataclass

from genai_otel_conformance.results import TestResult, merge_signal_counts
from genai_otel_conformance.specs import (
    DISPLAY_DEPRECATED_ATTRS,
    EVENT_TYPE_SPECS,
    METRIC_TYPE_SPECS,
    RequirementLevel,
    SignalTypeSpec,
    SPAN_TYPE_ORDER,
    SPAN_TYPE_SPECS,
)


@dataclass(frozen=True)
class RequirementLevelInfo:
    key: RequirementLevel
    label: str
    description: str


@dataclass(frozen=True)
class SignalTypeAttributeGroup:
    level: RequirementLevelInfo
    attrs: tuple[str, ...]


@dataclass(frozen=True)
class HeatmapColumn:
    header_text: str
    is_group_start: bool
    group_key: RequirementLevel | str = ""
    group_label: str = ""


@dataclass(frozen=True)
class HeatmapGroup:
    key: RequirementLevel
    label: str
    description: str
    colspan: int


def present_attributes(result: TestResult) -> set[str]:
    """Return all attribute names present in registry and non-registry stats."""
    attrs = set(result.observed.attrs)
    attrs.update(result.observed.non_registry_attrs)
    return attrs


_SIGNAL_TYPE_LEVELS = (
    RequirementLevelInfo(RequirementLevel.REQUIRED, "Required", "Must be present for spans of this type."),
    RequirementLevelInfo(
        RequirementLevel.CONDITIONALLY_REQUIRED,
        "Conditionally Required",
        "Required only when the span matches the relevant condition.",
    ),
    RequirementLevelInfo(RequirementLevel.RECOMMENDED, "Recommended", "Expected when the library exposes the signal."),
    RequirementLevelInfo(RequirementLevel.OPT_IN, "Opt-In", "Captured only when the user explicitly enables it."),
)


def _display_attrs_for_group(spec: SignalTypeSpec, level: RequirementLevel) -> tuple[str, ...]:
    """Return attrs for one visual group, including deprecated predecessors."""
    display_attrs: list[str] = []
    for attr in sorted(spec.attrs_for_requirement_level(level)):
        display_attrs.append(attr)
        deprecated_attr = DISPLAY_DEPRECATED_ATTRS.get(attr)
        if deprecated_attr is not None:
            display_attrs.append(deprecated_attr)
    return tuple(display_attrs)


def signal_type_attribute_groups(spec: SignalTypeSpec) -> list[SignalTypeAttributeGroup]:
    """Return ordered attribute groups for a signal-type specification."""
    groups: list[SignalTypeAttributeGroup] = []
    for level in _SIGNAL_TYPE_LEVELS:
        attrs = _display_attrs_for_group(spec, level.key)
        if attrs:
            groups.append(SignalTypeAttributeGroup(level, attrs))
    return groups


def signal_type_heatmap_columns(spec: SignalTypeSpec) -> list[HeatmapColumn]:
    """Return ordered heatmap columns for a signal-type specification."""
    return [
        HeatmapColumn(
            header_text=attr,
            is_group_start=i == 0,
            group_key=group.level.key,
            group_label=group.level.label,
        )
        for group in signal_type_attribute_groups(spec)
        for i, attr in enumerate(group.attrs)
    ]


def signal_type_heatmap_groups(spec: SignalTypeSpec) -> list[HeatmapGroup]:
    """Return grouped header metadata for a signal-type heatmap."""
    return [
        HeatmapGroup(
            key=group.level.key,
            label=group.level.label,
            description=group.level.description,
            colspan=len(group.attrs),
        )
        for group in signal_type_attribute_groups(spec)
    ]


def span_type_present_attributes(
    result: TestResult,
    span_type_key: str,
    level: RequirementLevel,
) -> set[str]:
    """Return attrs present for a span type at the requested requirement level."""
    all_present = present_attributes(result)
    return result.spans.per_type_attrs.get(span_type_key, all_present)


def relevant_span_type_keys(result: TestResult) -> list[str]:
    """Return span-type keys that are relevant for this result."""
    all_present = present_attributes(result)
    relevant: list[str] = []
    for span_type_key in SPAN_TYPE_ORDER:
        spec = SPAN_TYPE_SPECS[span_type_key]
        expected_attrs: list[str] = []
        for group in signal_type_attribute_groups(spec):
            expected_attrs.extend(group.attrs)
        if not expected_attrs:
            continue
        if spec.discriminator_attrs:
            if span_type_key in result.spans.detected_types:
                relevant.append(span_type_key)
        elif any(attr in all_present for attr in expected_attrs):
            relevant.append(span_type_key)
    return relevant


def build_statuses_from_present_names(
    expected_names: list[str],
    present_names: list[str] | set[str],
) -> dict[str, str]:
    """Expand a sparse present-name list into present/absent statuses."""
    present = set(present_names)
    return {
        name: "present" if name in present else "absent"
        for name in expected_names
    }


def build_span_type_present_names(result: TestResult) -> dict[str, list[str]]:
    """Return sparse per-span-type attribute lists for relevant span types."""
    sparse: dict[str, list[str]] = {}
    for span_type_key in relevant_span_type_keys(result):
        spec = SPAN_TYPE_SPECS[span_type_key]
        present_names: list[str] = []
        for group in signal_type_attribute_groups(spec):
            type_present = span_type_present_attributes(result, span_type_key, group.level.key)
            present_names.extend(
                attr for attr in group.attrs
                if attr in type_present
            )
        sparse[span_type_key] = present_names
    return sparse


def event_type_present_attributes(
    result: TestResult,
    event_name: str,
    level: RequirementLevel,
) -> set[str]:
    """Return attrs present for an event type at the requested requirement level."""
    all_present = present_attributes(result)
    if level is RequirementLevel.REQUIRED:
        return result.detected.event_attrs.get(event_name, all_present)
    return result.detected.event_any_attrs.get(event_name, all_present)


def metric_type_present_attributes(
    result: TestResult,
    metric_name: str,
    level: RequirementLevel,
) -> set[str]:
    """Return attrs present for a metric type at the requested requirement level."""
    all_present = present_attributes(result)
    if level is RequirementLevel.REQUIRED:
        return result.detected.metric_attrs.get(metric_name, all_present)
    return result.detected.metric_any_attrs.get(metric_name, all_present)


def _build_signal_type_present_names(
    signal_type_specs: dict[str, SignalTypeSpec],
    merged_counts: dict[str, int],
    present_fn: callable,
) -> dict[str, list[str]]:
    """Return sparse per-signal-type attribute lists for detected signals."""
    sparse: dict[str, list[str]] = {}
    for signal_name, spec in signal_type_specs.items():
        if merged_counts.get(signal_name, 0) <= 0:
            continue
        present_names: list[str] = []
        for group in signal_type_attribute_groups(spec):
            type_present = present_fn(signal_name, group.level.key)
            present_names.extend(
                attr for attr in group.attrs
                if attr in type_present
            )
        sparse[signal_name] = present_names
    return sparse


def build_event_type_present_names(result: TestResult) -> dict[str, list[str]]:
    """Return sparse per-event-type attribute lists for detected events."""
    merged = merge_signal_counts(result.observed.events, result.detected.events)
    return _build_signal_type_present_names(
        EVENT_TYPE_SPECS,
        merged,
        lambda name, level: event_type_present_attributes(result, name, level),
    )


def build_metric_type_present_names(result: TestResult) -> dict[str, list[str]]:
    """Return sparse per-metric-type attribute lists for detected metrics."""
    merged = merge_signal_counts(result.observed.metrics, result.detected.metrics)
    return _build_signal_type_present_names(
        METRIC_TYPE_SPECS,
        merged,
        lambda name, level: metric_type_present_attributes(result, name, level),
    )