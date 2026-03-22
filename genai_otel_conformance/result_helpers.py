from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class ResultLike(Protocol):
    seen_attrs: Mapping[str, int]
    seen_non_registry_attrs: Mapping[str, int]
    detected_span_types: set[str]
    per_type_attrs: Mapping[str, set[str]]


def merge_signal_counts(
    statistics_counts: Mapping[str, int],
    detected_counts: Mapping[str, int],
) -> dict[str, int]:
    """Merge statistic-derived and sample-derived signal counts."""
    merged = dict(statistics_counts)
    for name, count in detected_counts.items():
        merged[name] = max(merged.get(name, 0), count)
    return merged


def present_attributes(result: ResultLike) -> set[str]:
    """Return all attribute names present in registry and non-registry stats."""
    return set(result.seen_attrs) | set(result.seen_non_registry_attrs)


def span_type_present_attributes(result: ResultLike, span_type_key: str) -> set[str]:
    """Return attrs present for a span type, falling back to global presence."""
    all_present = present_attributes(result)
    return result.per_type_attrs.get(span_type_key, all_present)


def _expected_span_type_attributes(spec: Mapping[str, Any]) -> list[str]:
    attrs: list[str] = []
    for level in ("required", "conditionally_required", "recommended"):
        attrs.extend(spec.get(level, []))
    return attrs


def _is_relevant_span_type(
    result: ResultLike,
    span_type_key: str,
    spec: Mapping[str, Any],
) -> bool:
    all_present = present_attributes(result)
    discriminators = spec.get("discriminator_attrs", set())
    if discriminators:
        return bool(all_present & discriminators) or span_type_key in result.detected_span_types
    return bool(all_present & set(_expected_span_type_attributes(spec)))


def relevant_span_type_keys(
    result: ResultLike,
    span_type_order: Sequence[str],
    span_type_specs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return span-type keys that are relevant for this result."""
    relevant: list[str] = []
    for span_type_key in span_type_order:
        spec = span_type_specs[span_type_key]
        if not _expected_span_type_attributes(spec):
            continue
        if _is_relevant_span_type(result, span_type_key, spec):
            relevant.append(span_type_key)
    return relevant


def build_signal_statuses(
    signal_names: Sequence[str],
    statistics_counts: Mapping[str, int],
    detected_counts: Mapping[str, int],
) -> dict[str, str]:
    """Return present/absent statuses for the given signal names."""
    merged_counts = merge_signal_counts(statistics_counts, detected_counts)
    return {
        name: ("present" if merged_counts.get(name, 0) > 0 else "absent")
        for name in signal_names
    }


def build_span_type_statuses(
    result: ResultLike,
    span_type_order: Sequence[str],
    span_type_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Return present/absent attribute statuses for relevant span types."""
    statuses: dict[str, dict[str, str]] = {}
    for span_type_key in relevant_span_type_keys(result, span_type_order, span_type_specs):
        spec = span_type_specs[span_type_key]
        type_present = span_type_present_attributes(result, span_type_key)
        statuses[span_type_key] = {
            attr: ("present" if attr in type_present else "absent")
            for attr in _expected_span_type_attributes(spec)
        }
    return statuses