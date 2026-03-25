#!/usr/bin/env python3
"""Generate a static HTML conformance dashboard.

Usage:
    python generate_dashboard.py [--output-dir DIR]

Builds index.html from committed tests/<lang>/<lib>/data-<eco>.json files.
When local Weaver output exists under tests/<lang>/<lib>/results/<eco>/, the
script also generates details.html and local-only event/metric heatmaps.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.metadata import (
    ECOSYSTEM_REPOS,
    NATIVE_REPOS,
    extract_version_from_deps,
)
from genai_otel_conformance.statuses import (
    HeatmapColumn,
    HeatmapGroup,
    merge_signal_counts,
    relevant_span_type_keys,
    span_type_attribute_groups,
    span_type_heatmap_groups,
    span_type_heatmap_columns,
    span_type_present_attributes,
)
from genai_otel_conformance.results import (
    TestResult,
    parse_all_results,
)
from genai_otel_conformance.specs import (
    DISPLAY_DEPRECATED_ATTRS,
    GENAI_EVENT_TYPES,
    GENAI_METRIC_TYPES,
    SPAN_TYPE_ORDER,
    SPAN_TYPE_SPECS,
)
from genai_otel_conformance.data_files import (
    TestDataEntry,
    load_test_data_files,
    make_anchor_id,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


# ── HTML generation ──────────────────────────────────────────────────


def _compute_rowspans(rows: list[dict]) -> None:
    """Compute lib_rowspan / lang_rowspan for Library → Language hierarchy."""
    for row in rows:
        row["lang_rowspan"] = 0
        row["lib_rowspan"] = 0
    for _, lib_group in groupby(rows, key=lambda r: r["lib_display"]):
        lib_rows = list(lib_group)
        lib_rows[0]["lib_rowspan"] = len(lib_rows)
        lang_offset = 0
        for _, lang_group in groupby(lib_rows, key=lambda r: r["language"]):
            lang_rows = list(lang_group)
            lib_rows[lang_offset]["lang_rowspan"] = len(lang_rows)
            lang_offset += len(lang_rows)


_LANGUAGE_ORDER = {"python": 0, "js": 1, "java": 2, "c#": 3}


def _language_order(language: str) -> int:
    return _LANGUAGE_ORDER.get(language.lower(), 99)


def _test_entry_sort_key(entry: TestDataEntry) -> tuple[str, int, str]:
    return (
        entry.library_display.lower(),
        _language_order(entry.language_display),
        entry.ecosystem_display.lower(),
    )


def _build_heatmap(
    label: str,
    columns: list[HeatmapColumn],
    entries: list[TestDataEntry],
    details_available: bool,
    status_extractor: Callable[[TestDataEntry], dict[str, str]],
    deprecated_attrs: set[str] | None = None,
    column_groups: list[HeatmapGroup] | None = None,
) -> dict | None:
    if not entries or not columns:
        return None

    rows = []
    for entry in sorted(entries, key=_test_entry_sort_key):
        rows.append({
            "test_name": entry.test_name,
            "has_details": details_available,
            "lib_display": entry.library_display,
            "language": entry.language_display,
            "eco_display": entry.ecosystem_display,
            "instrumentation_version": extract_version_from_deps(
                entry.lang,
                entry.library,
                entry.ecosystem,
            ),
            "cells": _build_status_cells(columns, status_extractor(entry), deprecated_attrs),
        })

    _compute_rowspans(rows)
    return {
        "label": label,
        "columns": columns,
        "column_groups": column_groups or [],
        "rows": rows,
    }


def _build_status_cells(
    columns: list[HeatmapColumn],
    statuses: dict[str, str],
    deprecated_attrs: set[str] | None = None,
) -> list[dict]:
    deprecated = deprecated_attrs or set()
    cells = []
    for col in columns:
        name = col.header_text
        is_group_start = col.is_group_start
        present = statuses.get(name) == "present"
        cls = ("deprecated" if name in deprecated else "present") if present else "absent"
        if is_group_start:
            cls += " group-start"
        cells.append({"cls": cls, "symbol": "\u2713" if present else ""})
    return cells


def _signal_columns(signal_names: list[str]) -> list[HeatmapColumn]:
    return [
        HeatmapColumn(header_text=name, is_group_start=i == 0)
        for i, name in enumerate(signal_names)
    ]


def _detail_repo(lang_slug: str, library: str, ecosystem: str, language: str) -> str:
    if ecosystem == "native":
        return NATIVE_REPOS.get((lang_slug, library), "")
    return ECOSYSTEM_REPOS.get((ecosystem, language), "")


def _entity_summary(result: TestResult) -> str:
    parts = [
        f"{count} {t}{'s' if count != 1 else ''}"
        for t in ("span", "log", "resource", "attribute")
        if (count := result.observed.entity_counts.get(t, 0)) > 0
    ]
    if parts:
        return ", ".join(parts)
    if result.statistics is not None and result.statistics.get("total_entities") == 0:
        return "0 entities"
    return ""


def _build_span_sections(result: TestResult) -> list[dict]:
    sections = []
    for span_type_key in relevant_span_type_keys(result):
        spec = SPAN_TYPE_SPECS[span_type_key]
        groups = []
        for group_spec in span_type_attribute_groups(spec):
            type_present = span_type_present_attributes(result, span_type_key, group_spec.key)
            attrs = []
            for attr in group_spec.attrs:
                if attr in type_present:
                    count = result.observed.attrs.get(attr, result.observed.non_registry_attrs.get(attr, 0))
                    attrs.append({"name": attr, "present": True, "count": count})
                else:
                    attrs.append({"name": attr, "present": False, "count": 0})
            groups.append({"label": group_spec.label, "attrs": attrs})
        sections.append({"label": spec.label, "groups": groups})
    return sections


def _sorted_count_items(counts: dict[str, int]) -> list[dict[str, int | str]]:
    return [{"name": name, "count": count} for name, count in sorted(counts.items())]


def _build_detail(
    anchor_id: str,
    label: str,
    lang_slug: str,
    library: str,
    ecosystem: str,
    language: str,
    result: TestResult | None,
) -> dict:
    detail: dict = {
        "test_name": anchor_id,
        "label": label,
        "has_local_run": result is not None,
        "has_data": result is not None and result.observed.has_data,
        "has_empty_run": result is not None and result.statistics is not None and not result.observed.has_data,
        "violation_count": result.violation_count if result else 0,
        "instrumentation_version": extract_version_from_deps(lang_slug, library, ecosystem),
        "repo": _detail_repo(lang_slug, library, ecosystem, language),
        "entity_summary": _entity_summary(result) if result else "",
        "span_sections": [],
        "non_registry_attrs": [],
        "events": [],
        "metrics": [],
        "violation_messages": result.violation_messages if result and result.violation_messages else [],
    }

    if result and result.observed.has_data:
        detail["span_sections"] = _build_span_sections(result)

        if result.observed.non_registry_attrs:
            detail["non_registry_attrs"] = _sorted_count_items(result.observed.non_registry_attrs)

        merged_events = merge_signal_counts(result.observed.events, result.detected.events)
        if merged_events:
            detail["events"] = _sorted_count_items(merged_events)

        merged_metrics = merge_signal_counts(result.observed.metrics, result.detected.metrics)
        if merged_metrics:
            detail["metrics"] = _sorted_count_items(merged_metrics)

    return detail


def _prepare_details(
    results: dict[str, TestResult],
    test_data_entries: list[TestDataEntry],
) -> list[dict]:
    """Prepare detailed result data for the template.

    Every known test (from committed data-*.json files) gets an anchor.
    Entries without local Weaver output show "Results not yet available."
    Results without a corresponding data file are appended at the end.
    """
    result_by_id: dict[str, TestResult] = {}
    for result in results.values():
        anchor_id = make_anchor_id(result.language, result.library, result.ecosystem)
        result_by_id[anchor_id] = result

    sorted_entries = sorted(test_data_entries, key=_test_entry_sort_key)
    seen_ids = {entry.test_name for entry in sorted_entries}

    # Build synthetic entries for results without a committed data file.
    extra_entries: list[TestDataEntry] = []
    for result in results.values():
        anchor_id = make_anchor_id(result.language, result.library, result.ecosystem)
        if anchor_id not in seen_ids:
            extra_entries.append(TestDataEntry.from_result(result))
    extra_entries.sort(key=_test_entry_sort_key)

    details = []
    for entry in sorted_entries + extra_entries:
        result = result_by_id.get(entry.test_name)
        details.append(
            _build_detail(
                entry.test_name,
                entry.label,
                entry.lang,
                entry.library,
                entry.ecosystem,
                entry.language_display,
                result,
            )
        )

    return details


def _prepare_heatmaps_from_data(
    test_data_entries: list[TestDataEntry],
    details_available: bool,
) -> list[dict]:
    """Build per-span-type heatmap data from loaded data-*.json entries."""
    heatmaps = []
    for st_key in SPAN_TYPE_ORDER:
        spec = SPAN_TYPE_SPECS[st_key]
        st_label = spec.label

        # Collect entries that have this span type.
        relevant = [entry for entry in test_data_entries if st_key in entry.spans]
        if not relevant:
            continue

        columns = span_type_heatmap_columns(spec)
        column_groups = span_type_heatmap_groups(spec)

        if not columns:
            continue

        heatmap = _build_heatmap(
            st_label,
            columns,
            relevant,
            details_available,
            lambda entry, _key=st_key: entry.spans[_key],
            deprecated_attrs=set(DISPLAY_DEPRECATED_ATTRS.values()),
            column_groups=column_groups,
        )
        if heatmap is not None:
            heatmaps.append(heatmap)

    return heatmaps


def _prepare_signal_heatmap(
    test_data_entries: list[TestDataEntry],
    signal_names: list[str],
    label: str,
    status_extractor: Callable[[TestDataEntry], dict[str, str]],
    details_available: bool = True,
) -> dict | None:
    """Build an event or metric heatmap from committed data-*.json entries."""
    relevant = [entry for entry in test_data_entries if status_extractor(entry)]

    columns = _signal_columns(signal_names)

    return _build_heatmap(label, columns, relevant, details_available, status_extractor)


def _has_result_directories() -> bool:
    """Return whether any local Weaver result directories exist."""
    for path in TESTS_DIR.glob("*/*/results/*"):
        if path.is_dir():
            return True
    return False


def _render_template(template_name: str, **context: object) -> str:
    """Render a dashboard template with shared environment setup."""
    import jinja2

    css = (TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template(template_name)
    return template.render(css=css, **context)


def generate_dashboard_html(test_data_entries: list[TestDataEntry], details_available: bool) -> str:
    """Generate the dashboard HTML with span heatmap tables.

    All dashboard heatmaps come from committed data-*.json files.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    heatmaps = _prepare_heatmaps_from_data(test_data_entries, details_available)
    event_heatmap = _prepare_signal_heatmap(
        test_data_entries,
        GENAI_EVENT_TYPES,
        "GenAI Events",
        lambda entry: entry.events,
        details_available,
    )
    metric_heatmap = _prepare_signal_heatmap(
        test_data_entries,
        GENAI_METRIC_TYPES,
        "GenAI Metrics",
        lambda entry: entry.metrics,
        details_available,
    )

    return _render_template(
        "dashboard.html",
        now=now,
        heatmaps=heatmaps,
        details_available=details_available,
        event_heatmap=event_heatmap,
        metric_heatmap=metric_heatmap,
    )


def generate_details_html(
    results: dict[str, TestResult],
    test_data_entries: list[TestDataEntry],
) -> str:
    """Generate the details HTML from Weaver results."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    details = _prepare_details(results, test_data_entries)
    return _render_template("details.html", now=now, details=details)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conformance dashboard")
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory to write output files (default: current directory)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = parse_all_results()
    if _has_result_directories() and not results:
        raise RuntimeError(
            "Found local Weaver result directories, but no parsable results. "
            "Clean tests/*/*/results or rerun the affected tests before "
            "generating the dashboard."
        )

    test_data_entries = load_test_data_files()
    # Details page has an anchor for every known test; entries without local
    # Weaver output show "Results not yet available."
    details_available = bool(test_data_entries)

    # Dashboard (index.html) always generated from data files only.
    dashboard_html = generate_dashboard_html(test_data_entries, details_available)
    (out / "index.html").write_text(dashboard_html, encoding="utf-8")
    print(f"Dashboard written to {out / 'index.html'}")

    # Details page always generated so all anchors exist for dashboard links.
    details_html = generate_details_html(results, test_data_entries)
    (out / "details.html").write_text(details_html, encoding="utf-8")
    print(f"Details written to {out / 'details.html'}")
    if results:
        print(f"  Tests with results: {len(results)}")


if __name__ == "__main__":
    main()
