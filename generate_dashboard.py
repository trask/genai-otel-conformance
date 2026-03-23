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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from genai_otel_conformance import TESTS_DIR
from genai_otel_conformance.metadata import (
    ECOSYSTEM_DISPLAY,
    ECOSYSTEM_REPOS,
    LANGUAGE_DISPLAY_NAMES,
    NATIVE_REPOS,
    extract_version_from_deps,
    library_display_name,
)
from genai_otel_conformance.statuses import (
    build_statuses_from_present_names,
    merge_signal_counts,
    relevant_span_type_keys,
    span_type_attribute_groups,
    span_type_heatmap_groups,
    span_type_heatmap_columns,
    span_type_present_attributes,
)
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
from genai_otel_conformance.locations import TestLocation

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"

_LANG_SLUG: dict[str, str] = {}
for _slug, _display in LANGUAGE_DISPLAY_NAMES.items():
    _LANG_SLUG[_display] = _slug


def _make_anchor_id(language: str, library: str, ecosystem: str) -> str:
    """Build an anchor ID in library-language-ecosystem order.

    This matches the dashboard's visual hierarchy: group by library,
    then by language, then by ecosystem.

    Args:
        language: Display language name (e.g. "Python", "JS").
        library: Library slug (e.g. "openai").
        ecosystem: Ecosystem slug (e.g. "otelcontrib").
    """
    lang_slug = _LANG_SLUG.get(language, language.lower())
    return f"{library}-{lang_slug}-{ecosystem}"

# Deprecated attributes: shown with yellow background when present.
DEPRECATED_ATTRS = {
    "gen_ai.system",
}

# ── Data structures ──────────────────────────────────────────────────


def parse_results() -> dict[str, TestResult]:
    """Parse all Weaver output directories under tests/.

    Layout: tests/<lang>/<lib>/results/<eco>/
    """
    results: dict[str, TestResult] = {}

    if not TESTS_DIR.exists():
        print(f"Tests directory not found: {TESTS_DIR}", file=sys.stderr)
        return results

    result_dirs = []
    for path in TESTS_DIR.glob("*/*/results/*"):
        if path.is_dir():
            result_dirs.append(path)

    for result_dir in sorted(result_dirs):
        location = TestLocation.from_results_dir(result_dir, TESTS_DIR)
        result = parse_result_dir(result_dir, location.test_name)
        if result is None:
            continue
        if not _has_detail_content(result):
            continue
        results[location.test_name] = result

    return results


def _has_detail_content(result: TestResult) -> bool:
    """Return whether a parsed result contains any renderable detail content."""
    return (
        result.statistics is not None
        or result.has_data
        or bool(result.violation_messages)
        or bool(result.seen_attrs)
        or bool(result.seen_non_registry_attrs)
        or bool(result.seen_events)
        or bool(result.detected_span_types)
        or bool(result.detected_events)
        or bool(result.detected_metrics)
    )


# ── HTML generation ──────────────────────────────────────────────────


def _compute_rowspans(rows: list[dict]) -> None:
    """Compute lib_rowspan / lang_rowspan for Library → Language hierarchy."""
    for row in rows:
        row["lang_rowspan"] = 0
        row["lib_rowspan"] = 0
    if not rows:
        return
    i = 0
    while i < len(rows):
        lib = rows[i]["lib_display"]
        lib_start = i
        while i < len(rows) and rows[i]["lib_display"] == lib:
            i += 1
        rows[lib_start]["lib_rowspan"] = i - lib_start
        j = lib_start
        while j < i:
            lang = rows[j]["language"]
            lang_start = j
            while j < i and rows[j]["language"] == lang:
                j += 1
            rows[lang_start]["lang_rowspan"] = j - lang_start


_LANGUAGE_ORDER = {"python": 0, "js": 1, "java": 2, "c#": 3}


def _language_order(language: str) -> int:
    return _LANGUAGE_ORDER.get(language.lower(), 99)


def _heatmap_sort_key(entry: dict) -> tuple[str, int, str]:
    return (
        entry.get("library", "").lower(),
        _language_order(entry.get("language", "")),
        entry.get("ecosystem", "").lower(),
    )


def _build_heatmap(
    label: str,
    columns: list[dict],
    entries: list[dict],
    details_available: bool,
    cell_builder,
    column_groups: list[dict] | None = None,
) -> dict | None:
    if not entries or not columns:
        return None

    rows = []
    for entry in sorted(entries, key=_heatmap_sort_key):
        rows.append({
            "test_name": entry.get("test_name", ""),
            "has_details": details_available,
            "lib_display": entry.get("library", ""),
            "language": entry.get("language", ""),
            "eco_display": entry.get("ecosystem", ""),
            "instrumentation_version": extract_version_from_deps(
                entry["_lang"],
                entry["_lib"],
                entry["_eco"],
            ),
            "cells": cell_builder(entry),
        })

    _compute_rowspans(rows)
    return {
        "label": label,
        "columns": columns,
        "column_groups": column_groups or [],
        "rows": rows,
    }


def _build_status_cells(
    definitions: list[tuple[str, bool]],
    statuses: dict[str, str],
    deprecated_attrs: set[str] | None = None,
) -> list[dict]:
    cells = []
    deprecated = deprecated_attrs or set()
    for name, is_group_start in definitions:
        status = statuses.get(name, "absent")
        if status == "present":
            cls = "deprecated" if name in deprecated else "present"
            symbol = "\u2713"
        else:
            cls = "absent"
            symbol = ""
        group_cls = " group-start" if is_group_start else ""
        cells.append({"cls": f"{cls}{group_cls}", "symbol": symbol})
    return cells


def _signal_columns(signal_names: list[str]) -> list[dict[str, object]]:
    columns: list[dict[str, object]] = []
    for index, signal_name in enumerate(signal_names):
        columns.append({"header_text": signal_name, "is_group_start": index == 0})
    return columns


def _column_definitions(columns: list[dict]) -> list[tuple[str, bool]]:
    definitions: list[tuple[str, bool]] = []
    for column in columns:
        definitions.append((column["header_text"], column["is_group_start"]))
    return definitions


def _detail_sort_key(entry: dict) -> tuple[str, int, str]:
    return _heatmap_sort_key(entry)


def _extra_result_sort_key(result: TestResult) -> tuple[str, str, str]:
    return (
        library_display_name(result.library).lower(),
        result.language.lower(),
        result.ecosystem.lower(),
    )


def _detail_repo(lang_slug: str, library: str, ecosystem: str, language: str) -> str:
    if ecosystem == "native":
        return NATIVE_REPOS.get((lang_slug, library), "")
    return ECOSYSTEM_REPOS.get((ecosystem, language), "")


def _entity_summary(result: TestResult) -> str:
    entity_parts = []
    for entity_type in ("span", "log", "resource", "attribute"):
        count = result.entity_counts.get(entity_type, 0)
        if count > 0:
            entity_parts.append(f"{count} {entity_type}{'s' if count != 1 else ''}")
    if entity_parts:
        return ", ".join(entity_parts)
    if result.statistics is not None and result.statistics.get("total_entities") == 0:
        return "0 entities"
    return ""


def _build_span_sections(result: TestResult) -> list[dict]:
    sections = []
    for span_type_key in relevant_span_type_keys(result, SPAN_TYPE_ORDER, SPAN_TYPE_SPECS):
        spec = SPAN_TYPE_SPECS[span_type_key]
        type_present = span_type_present_attributes(result, span_type_key)
        groups = []
        for group_spec in span_type_attribute_groups(spec):
            attrs = []
            for attr in group_spec["attrs"]:
                if attr in type_present:
                    count = result.seen_attrs.get(attr, result.seen_non_registry_attrs.get(attr, 0))
                    attrs.append({"name": attr, "present": True, "count": count})
                else:
                    attrs.append({"name": attr, "present": False, "count": 0})
            groups.append({"label": group_spec["label"], "attrs": attrs})
        sections.append({"label": spec["label"], "groups": groups})
    return sections


def _sorted_count_items(counts: dict[str, int]) -> list[dict[str, int | str]]:
    items: list[dict[str, int | str]] = []
    for name, count in sorted(counts.items()):
        items.append({"name": name, "count": count})
    return items


def _build_detail(
    anchor_id: str,
    label: str,
    lang_slug: str,
    library: str,
    ecosystem: str,
    language: str,
    result: TestResult | None,
) -> dict:
    has_local_run = result is not None
    has_data = has_local_run and result.has_data
    has_empty_run = has_local_run and result.statistics is not None and not result.has_data

    detail: dict = {
        "test_name": anchor_id,
        "label": label,
        "has_local_run": has_local_run,
        "has_data": has_data,
        "has_empty_run": has_empty_run,
        "violation_count": result.violation_count if result else 0,
        "instrumentation_version": extract_version_from_deps(lang_slug, library, ecosystem),
        "repo": _detail_repo(lang_slug, library, ecosystem, language),
        "entity_summary": "",
        "span_sections": [],
        "non_registry_attrs": [],
        "events": [],
        "metrics": [],
        "violation_messages": [],
    }

    if not result:
        return detail

    detail["entity_summary"] = _entity_summary(result)

    if result.has_data:
        detail["span_sections"] = _build_span_sections(result)

        if result.seen_non_registry_attrs:
            detail["non_registry_attrs"] = _sorted_count_items(result.seen_non_registry_attrs)

        merged_events = merge_signal_counts(result.seen_events, result.detected_events)
        if merged_events:
            detail["events"] = _sorted_count_items(merged_events)

        merged_metrics = merge_signal_counts(result.seen_metrics, result.detected_metrics)
        if merged_metrics:
            detail["metrics"] = _sorted_count_items(merged_metrics)

    if result.violation_messages:
        detail["violation_messages"] = result.violation_messages

    return detail


def _prepare_details(
    results: dict[str, TestResult],
    test_data_entries: list[dict],
) -> list[dict]:
    """Prepare detailed result data for the template.

    Every known test (from committed data-*.json files) gets an anchor.
    Entries without local Weaver output show "Results not yet available."
    """
    result_by_id: dict[str, TestResult] = {}
    for result in results.values():
        anchor_id = _make_anchor_id(result.language, result.library, result.ecosystem)
        result_by_id[anchor_id] = result

    sorted_entries = sorted(test_data_entries, key=_detail_sort_key)

    # Also include any results that have no corresponding data file.
    seen_ids: set[str] = set()
    for entry in sorted_entries:
        seen_ids.add(entry["test_name"])

    extra_results: list[TestResult] = []
    for result in results.values():
        anchor_id = _make_anchor_id(result.language, result.library, result.ecosystem)
        if anchor_id not in seen_ids:
            extra_results.append(result)
    extra_results.sort(key=_extra_result_sort_key)

    details = []

    for entry in sorted_entries:
        lang = entry["_lang"]
        lib = entry["_lib"]
        eco = entry["_eco"]
        language = entry.get("language", LANGUAGE_DISPLAY_NAMES.get(lang, lang))
        lib_display = entry.get("library", library_display_name(lib))
        eco_display = entry.get("ecosystem", ECOSYSTEM_DISPLAY.get(eco, eco))
        label = f"{lib_display} ({language}) \u2014 {eco_display}"
        result = result_by_id.get(entry["test_name"])
        details.append(_build_detail(entry["test_name"], label, lang, lib, eco, language, result))

    for result in extra_results:
        lang_slug = _LANG_SLUG.get(result.language, result.language.lower())
        anchor_id = _make_anchor_id(result.language, result.library, result.ecosystem)
        lib_display = library_display_name(result.library)
        eco_display = ECOSYSTEM_DISPLAY.get(result.ecosystem, result.ecosystem)
        label = f"{lib_display} ({result.language}) \u2014 {eco_display}"
        details.append(
            _build_detail(
                anchor_id,
                label,
                lang_slug,
                result.library,
                result.ecosystem,
                result.language,
                result,
            )
        )

    return details


def _load_test_data_files() -> list[dict]:
    """Discover and load committed data-*.json files from tests/.

    Only files in the canonical tests/<lang>/<lib>/data-<ecosystem>.json layout
    are considered. This avoids pulling in copied workspace artifacts such as
    tests/js/node_modules/*/data-*.json.

    Returns a list of loaded data dicts (each augmented with the file path).
    """
    entries: list[dict] = []
    if not TESTS_DIR.is_dir():
        return entries
    for data_file in sorted(TESTS_DIR.glob("*/*/data-*.json")):
        try:
            data = json.loads(data_file.read_text(encoding="utf-8"))
            location = TestLocation.from_data_file(data_file, TESTS_DIR)
            data.setdefault("library", library_display_name(location.library))
            data.setdefault("language", LANGUAGE_DISPLAY_NAMES.get(location.lang, location.lang))
            data.setdefault(
                "ecosystem",
                ECOSYSTEM_DISPLAY.get(location.ecosystem, location.ecosystem),
            )
            data["test_name"] = _make_anchor_id(data["language"], location.library, location.ecosystem)
            data["_lang"] = location.lang
            data["_lib"] = location.library
            data["_eco"] = location.ecosystem
            entries.append(_normalize_test_data_entry(data))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {data_file}: {e}", file=sys.stderr)
    return entries


def _normalize_signal_data(
    value: object,
    signal_names: list[str],
) -> dict[str, str]:
    return build_statuses_from_present_names(signal_names, _present_signal_names(value))


def _present_name_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [name for name in value if isinstance(name, str)]


def _present_signal_names(value: object) -> list[str]:
    if isinstance(value, dict):
        return [name for name in value if isinstance(name, str)]
    return _present_name_list(value)


def _normalize_span_type_data(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for span_type_key, spec in SPAN_TYPE_SPECS.items():
        if span_type_key not in value:
            continue

        expected_names = [column["header_text"] for column in span_type_heatmap_columns(spec)]
        present_names = _present_name_list(value[span_type_key])
        normalized[span_type_key] = build_statuses_from_present_names(expected_names, present_names)

    return normalized


def _raw_spans(entry: dict) -> object:
    if "spans" in entry:
        return entry.get("spans")
    return entry.get("span_types")


def _normalize_test_data_entry(entry: dict) -> dict:
    normalized = dict(entry)
    normalized["events"] = _normalize_signal_data(entry.get("events"), GENAI_EVENT_TYPES)
    normalized["metrics"] = _normalize_signal_data(entry.get("metrics"), GENAI_METRIC_TYPES)
    normalized["spans"] = _normalize_span_type_data(_raw_spans(entry))
    normalized.pop("span_types", None)
    return normalized


def _build_span_type_cells(
    entry: dict,
    span_type_key: str,
    definitions: list[tuple[str, bool]],
) -> list[dict]:
    return _build_status_cells(
        definitions,
        entry["spans"][span_type_key],
        deprecated_attrs=DEPRECATED_ATTRS,
    )


def _build_signal_cells(
    entry: dict,
    data_key: str,
    definitions: list[tuple[str, bool]],
) -> list[dict]:
    return _build_status_cells(definitions, entry.get(data_key, {}))


def _entries_with_span_type(test_data_entries: list[dict], span_type_key: str) -> list[dict]:
    entries: list[dict] = []
    for entry in test_data_entries:
        spans = entry.get("spans", {})
        if span_type_key in spans:
            entries.append(entry)
    return entries


def _entries_with_key(test_data_entries: list[dict], data_key: str) -> list[dict]:
    entries: list[dict] = []
    for entry in test_data_entries:
        if data_key in entry:
            entries.append(entry)
    return entries


def _prepare_heatmaps_from_data(
    test_data_entries: list[dict],
    details_available: bool,
) -> list[dict]:
    """Build per-span-type heatmap data from loaded data-*.json entries."""
    heatmaps = []
    for st_key in SPAN_TYPE_ORDER:
        spec = SPAN_TYPE_SPECS[st_key]
        st_label = spec["label"]

        # Collect entries that have this span type.
        relevant = _entries_with_span_type(test_data_entries, st_key)
        if not relevant:
            continue

        columns = span_type_heatmap_columns(spec)
        column_groups = span_type_heatmap_groups(spec)
        col_defs = _column_definitions(columns)

        if not columns:
            continue

        def build_cells(entry: dict) -> list[dict]:
            return _build_span_type_cells(entry, st_key, col_defs)

        heatmap = _build_heatmap(
            st_label,
            columns,
            relevant,
            details_available,
            build_cells,
            column_groups=column_groups,
        )
        if heatmap is not None:
            heatmaps.append(heatmap)

    return heatmaps


def _prepare_signal_heatmap(
    test_data_entries: list[dict],
    signal_names: list[str],
    label: str,
    data_key: str,
    details_available: bool = True,
) -> dict | None:
    """Build an event or metric heatmap from committed data-*.json entries."""
    relevant = _entries_with_key(test_data_entries, data_key)

    columns = _signal_columns(signal_names)
    definitions = _column_definitions(columns)

    def build_cells(entry: dict) -> list[dict]:
        return _build_signal_cells(entry, data_key, definitions)

    return _build_heatmap(label, columns, relevant, details_available, build_cells)


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


def generate_dashboard_html(test_data_entries: list[dict], details_available: bool) -> str:
    """Generate the dashboard HTML with span heatmap tables.

    All dashboard heatmaps come from committed data-*.json files.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    heatmaps = _prepare_heatmaps_from_data(test_data_entries, details_available)
    event_heatmap = _prepare_signal_heatmap(
        test_data_entries,
        GENAI_EVENT_TYPES,
        "GenAI Events",
        "events",
        details_available,
    )
    metric_heatmap = _prepare_signal_heatmap(
        test_data_entries,
        GENAI_METRIC_TYPES,
        "GenAI Metrics",
        "metrics",
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
    test_data_entries: list[dict],
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

    results = parse_results()
    if _has_result_directories() and not results:
        raise RuntimeError(
            "Found local Weaver result directories, but no parsable results. "
            "Clean tests/*/*/results or rerun the affected tests before "
            "generating the dashboard."
        )

    test_data_entries = _load_test_data_files()
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
