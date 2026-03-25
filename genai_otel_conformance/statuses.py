"""Build heatmap statuses and attribute lookups from test results."""

from __future__ import annotations

from typing import NamedTuple

from genai_otel_conformance.results import TestResult
from genai_otel_conformance.specs import DISPLAY_DEPRECATED_ATTRS, SPAN_TYPE_ORDER, SPAN_TYPE_SPECS


class SpanTypeLevel(NamedTuple):
    key: str
    label: str
    description: str


def merge_signal_counts(
    statistics_counts: dict[str, int],
    detected_counts: dict[str, int],
) -> dict[str, int]:
    """Merge statistic-derived and sample-derived signal counts."""
    merged = dict(statistics_counts)
    for name, count in detected_counts.items():
        merged[name] = max(merged.get(name, 0), count)
    return merged


def present_attributes(result: TestResult) -> set[str]:
    """Return all attribute names present in registry and non-registry stats."""
    attrs = set(result.observed.attrs)
    attrs.update(result.observed.non_registry_attrs)
    return attrs


_SPAN_TYPE_LEVELS = (
    SpanTypeLevel("required", "Required", "Must be present for spans of this type."),
    SpanTypeLevel(
        "conditionally_required",
        "Conditionally Required",
        "Required only when the span matches the relevant condition.",
    ),
    SpanTypeLevel("recommended", "Recommended", "Expected when the library exposes the signal."),
    SpanTypeLevel("opt_in", "Opt-In", "Captured only when the user explicitly enables it."),
)


def _display_attrs_for_group(spec: dict, level: str) -> list[str]:
    """Return attrs for one visual group, including deprecated predecessors."""
    display_attrs: list[str] = []
    for attr in sorted(spec.get(level, [])):
        display_attrs.append(attr)
        deprecated_attr = DISPLAY_DEPRECATED_ATTRS.get(attr)
        if deprecated_attr is not None:
            display_attrs.append(deprecated_attr)
    return display_attrs


def span_type_attribute_groups(spec: dict) -> list[dict[str, object]]:
    """Return ordered attribute groups for a span-type specification."""
    groups: list[dict[str, object]] = []
    for level in _SPAN_TYPE_LEVELS:
        attrs = _display_attrs_for_group(spec, level.key)
        if attrs:
            groups.append({
                "key": level.key,
                "label": level.label,
                "description": level.description,
                "attrs": attrs,
            })
    return groups


def span_type_heatmap_columns(spec: dict) -> list[dict[str, object]]:
    """Return ordered heatmap columns for a span-type specification."""
    return [
        {
            "header_text": attr,
            "is_group_start": i == 0,
            "group_key": group["key"],
            "group_label": group["label"],
        }
        for group in span_type_attribute_groups(spec)
        for i, attr in enumerate(group["attrs"])
    ]


def span_type_heatmap_groups(spec: dict) -> list[dict[str, object]]:
    """Return grouped header metadata for a span-type heatmap."""
    return [
        {
            "key": group["key"],
            "label": group["label"],
            "description": group["description"],
            "colspan": len(group["attrs"]),
        }
        for group in span_type_attribute_groups(spec)
    ]


def span_type_present_attributes(
    result: TestResult,
    span_type_key: str,
    level: str,
) -> set[str]:
    """Return attrs present for a span type at the requested requirement level."""
    all_present = present_attributes(result)
    if level == "required":
        return result.spans.per_type_attrs.get(span_type_key, all_present)
    return result.spans.per_type_any_attrs.get(span_type_key, all_present)


def _expected_span_type_attributes(spec: dict) -> list[str]:
    attrs: list[str] = []
    for group in span_type_attribute_groups(spec):
        attrs.extend(group["attrs"])
    return attrs


def _is_relevant_span_type(
    result: TestResult,
    span_type_key: str,
    spec: dict,
) -> bool:
    if spec.get("discriminator_attrs"):
        return span_type_key in result.spans.detected_types

    all_present = present_attributes(result)
    for attr in _expected_span_type_attributes(spec):
        if attr in all_present:
            return True
    return False


def relevant_span_type_keys(result: TestResult) -> list[str]:
    """Return span-type keys that are relevant for this result."""
    relevant: list[str] = []
    for span_type_key in SPAN_TYPE_ORDER:
        spec = SPAN_TYPE_SPECS[span_type_key]
        if not _expected_span_type_attributes(spec):
            continue
        if _is_relevant_span_type(result, span_type_key, spec):
            relevant.append(span_type_key)
    return relevant


def build_present_signal_entries(
    signal_names: list[str],
    statistics_counts: dict[str, int],
    detected_counts: dict[str, int],
) -> dict[str, list[str]]:
    """Return sparse per-signal attribute lists for observed signals."""
    merged_counts = merge_signal_counts(statistics_counts, detected_counts)
    return {
        name: []
        for name in signal_names
        if merged_counts.get(name, 0) > 0
    }


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
        for group in span_type_attribute_groups(spec):
            type_present = span_type_present_attributes(result, span_type_key, group["key"])
            present_names.extend(
                attr for attr in group["attrs"]
                if attr in type_present
            )
        sparse[span_type_key] = present_names
    return sparse