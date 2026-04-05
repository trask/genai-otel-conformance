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
    relevant_span_type_keys,
    signal_type_attribute_groups,
    signal_type_heatmap_columns,
    signal_type_heatmap_groups,
    span_type_present_attributes,
)
from genai_otel_conformance.results import (
    TestResult,
    merge_signal_counts,
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
from genai_otel_conformance.views import (
    CountItemView,
    DetailAttributeView,
    DetailGroupView,
    DetailView,
    HeatmapRow,
    HeatmapView,
    SpanSectionView,
    StatusCell,
)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


# ── HTML generation ──────────────────────────────────────────────────


def _compute_rowspans(rows: list[HeatmapRow]) -> None:
    """Compute lib_rowspan / lang_rowspan for Library → Language hierarchy."""
    for _, lib_group in groupby(rows, key=lambda row: row.lib_display):
        lib_rows = list(lib_group)
        lib_rows[0].lib_rowspan = len(lib_rows)
        lang_offset = 0
        for _, lang_group in groupby(lib_rows, key=lambda row: row.language):
            lang_rows = list(lang_group)
            lib_rows[lang_offset].lang_rowspan = len(lang_rows)
            lang_offset += len(lang_rows)


_LANGUAGE_ORDER = {"python": 0, "js": 1, "java": 2, "c#": 3}


def _test_entry_sort_key(entry: TestDataEntry) -> tuple[str, int, int, str]:
    return (
        entry.library_display.lower(),
        _LANGUAGE_ORDER.get(entry.language_display.lower(), 99),
        0 if entry.ecosystem == "prototype" else 1 if entry.ecosystem == "otelcontrib" else 2,
        entry.ecosystem_display.lower(),
    )


def _build_heatmap(
    label: str,
    anchor_id: str,
    columns: list[HeatmapColumn],
    entries: list[TestDataEntry],
    details_available: bool,
    status_extractor: Callable[[TestDataEntry], dict[str, str]],
    deprecated_attrs: set[str] | None = None,
    column_groups: list[HeatmapGroup] | None = None,
) -> HeatmapView | None:
    if not columns:
        return None

    rows: list[HeatmapRow] = []
    for entry in sorted(entries, key=_test_entry_sort_key):
        rows.append(HeatmapRow(
            test_name=entry.test_name,
            has_details=details_available,
            lib_display=entry.library_display,
            language=entry.language_display,
            eco_display=entry.ecosystem_display,
            instrumentation_version=extract_version_from_deps(
                entry.lang,
                entry.library,
                entry.ecosystem,
            ),
            cells=_build_status_cells(columns, status_extractor(entry), deprecated_attrs),
        ))

    _compute_rowspans(rows)
    return HeatmapView(label, anchor_id, columns, column_groups or [], rows)


def _build_status_cells(
    columns: list[HeatmapColumn],
    statuses: dict[str, str],
    deprecated_attrs: set[str] | None = None,
) -> list[StatusCell]:
    deprecated = deprecated_attrs or set()
    cells: list[StatusCell] = []
    for col in columns:
        present = statuses.get(col.header_text) == "present"
        cls = ("deprecated" if col.header_text in deprecated else "present") if present else "absent"
        if col.is_group_start:
            cls += " group-start"
        cells.append(StatusCell(cls, "\u2713" if present else ""))
    return cells


def _detail_repo(entry: TestDataEntry) -> str:
    if entry.ecosystem == "native":
        return NATIVE_REPOS.get((entry.lang, entry.library), "")
    return ECOSYSTEM_REPOS.get((entry.ecosystem, entry.language_display), "")


def _entity_summary(result: TestResult) -> str:
    parts: list[str] = []
    for t in ("span", "log", "resource", "attribute"):
        count = result.observed.entity_counts.get(t, 0)
        if count > 0:
            suffix = "s" if count != 1 else ""
            parts.append(f"{count} {t}{suffix}")
    if parts:
        return ", ".join(parts)
    if result.statistics is not None and result.statistics.get("total_entities") == 0:
        return "0 entities"
    return ""


def _build_span_sections(result: TestResult) -> list[SpanSectionView]:
    sections: list[SpanSectionView] = []
    for span_type_key in relevant_span_type_keys(result):
        spec = SPAN_TYPE_SPECS[span_type_key]
        groups: list[DetailGroupView] = []
        for group_spec in signal_type_attribute_groups(spec):
            type_present = span_type_present_attributes(result, span_type_key, group_spec.level.key)
            attrs: list[DetailAttributeView] = []
            for attr in group_spec.attrs:
                if attr in type_present:
                    count = result.observed.attrs.get(attr, result.observed.non_registry_attrs.get(attr, 0))
                    attrs.append(DetailAttributeView(attr, True, count))
                else:
                    attrs.append(DetailAttributeView(attr, False, 0))
            groups.append(DetailGroupView(group_spec.level.label, attrs))
        sections.append(SpanSectionView(spec.label, groups))
    return sections


def _sorted_count_items(counts: dict[str, int]) -> list[CountItemView]:
    return [CountItemView(name, count) for name, count in sorted(counts.items())]


def _build_detail(
    entry: TestDataEntry,
    result: TestResult | None,
) -> DetailView:
    span_sections: list[SpanSectionView] = []
    non_registry_attrs: list[CountItemView] = []
    metrics: list[CountItemView] = []
    events: list[CountItemView] = []

    if result and result.observed.has_data:
        span_sections = _build_span_sections(result)

        if result.observed.non_registry_attrs:
            non_registry_attrs = _sorted_count_items(result.observed.non_registry_attrs)

        merged_metrics = merge_signal_counts(result.observed.metrics, result.detected.metrics)
        if merged_metrics:
            metrics = _sorted_count_items(merged_metrics)

        merged_events = merge_signal_counts(result.observed.events, result.detected.events)
        if merged_events:
            events = _sorted_count_items(merged_events)

    return DetailView(
        test_name=entry.test_name,
        label=entry.label,
        has_local_run=result is not None,
        has_data=result is not None and result.observed.has_data,
        has_empty_run=result is not None and result.statistics is not None and not result.observed.has_data,
        violation_count=result.violation_count if result else 0,
        instrumentation_version=extract_version_from_deps(entry.lang, entry.library, entry.ecosystem),
        repo=_detail_repo(entry),
        entity_summary=_entity_summary(result) if result else "",
        span_sections=span_sections,
        non_registry_attrs=non_registry_attrs,
        metrics=metrics,
        events=events,
        violation_messages=result.violation_messages if result else [],
    )


def _prepare_details(
    results: dict[str, TestResult],
    test_data_entries: list[TestDataEntry],
) -> list[DetailView]:
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
    seen_ids = {entry.test_name for entry in test_data_entries}

    # Build synthetic entries for results without a committed data file.
    extra_entries: list[TestDataEntry] = []
    for anchor_id, result in result_by_id.items():
        if anchor_id not in seen_ids:
            extra_entries.append(TestDataEntry.from_result(result))
    extra_entries.sort(key=_test_entry_sort_key)

    details: list[DetailView] = []
    for entry in sorted_entries + extra_entries:
        result = result_by_id.get(entry.test_name)
        details.append(_build_detail(entry, result))

    return details


def _prepare_heatmaps_from_data(
    test_data_entries: list[TestDataEntry],
    details_available: bool,
) -> list[HeatmapView]:
    """Build per-span-type heatmap data from loaded data-*.json entries."""
    heatmaps: list[HeatmapView] = []
    deprecated_attrs = set(DISPLAY_DEPRECATED_ATTRS.values())
    for st_key in SPAN_TYPE_ORDER:
        spec = SPAN_TYPE_SPECS[st_key]

        # Collect entries that have this span type.
        relevant = [entry for entry in test_data_entries if st_key in entry.spans]
        if not relevant:
            continue

        columns = signal_type_heatmap_columns(spec)
        column_groups = signal_type_heatmap_groups(spec)

        if not columns:
            continue

        anchor_id = f"span-{st_key.replace('_', '-')}"
        heatmap = _build_heatmap(
            f"{spec.label} Spans",
            anchor_id,
            columns,
            relevant,
            details_available,
            lambda entry, _key=st_key: entry.spans[_key],
            deprecated_attrs=deprecated_attrs,
            column_groups=column_groups,
        )
        if heatmap is not None:
            heatmaps.append(heatmap)

    return heatmaps


def _prepare_individual_signal_heatmaps(
    test_data_entries: list[TestDataEntry],
    signal_names: list[str],
    anchor_prefix: str,
    entry_statuses: Callable[[TestDataEntry], dict[str, str]],
    details_available: bool,
) -> list[HeatmapView]:
    """Build one heatmap per signal name (event type or metric type)."""
    relevant = [entry for entry in test_data_entries if entry_statuses(entry)]
    if not relevant:
        return []

    heatmaps: list[HeatmapView] = []
    for name in signal_names:
        anchor_id = f"{anchor_prefix}-{name.replace('.', '-').replace('_', '-')}"
        columns = [HeatmapColumn(header_text="Present", is_group_start=True)]
        heatmap = _build_heatmap(
            name,
            anchor_id,
            columns,
            relevant,
            details_available,
            lambda entry, _n=name: {"Present": entry_statuses(entry).get(_n, "absent")},
        )
        if heatmap is not None:
            heatmaps.append(heatmap)
    return heatmaps


def _has_result_directories() -> bool:
    """Return whether any local Weaver result directories exist."""
    return any(TESTS_DIR.glob("*/*/results/*"))


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
    event_heatmaps = _prepare_individual_signal_heatmaps(
        test_data_entries,
        GENAI_EVENT_TYPES,
        "event",
        lambda entry: entry.events,
        details_available,
    )
    metric_heatmaps = _prepare_individual_signal_heatmaps(
        test_data_entries,
        GENAI_METRIC_TYPES,
        "metric",
        lambda entry: entry.metrics,
        details_available,
    )

    return _render_template(
        "dashboard.html",
        now=now,
        heatmaps=heatmaps + metric_heatmaps + event_heatmaps,
        details_available=details_available,
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
