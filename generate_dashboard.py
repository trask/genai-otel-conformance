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

from run_test import (
    _LANG_DIRS,
    ECOSYSTEM_DISPLAY,
    ECOSYSTEM_REPOS,
    GENAI_EVENT_TYPES,
    GENAI_METRIC_TYPES,
    SPAN_TYPE_ORDER,
    SPAN_TYPE_SPECS,
    TestResult,
    extract_version_from_deps,
    parse_result_dir,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"
TESTS_DIR = SCRIPT_DIR / "tests"

# ── Library display names / native repos (from tests/<language>/<library>/metadata.json) ─

# Reverse of _LANG_DIRS: display language → directory slug.
_LANG_SLUG = {v: k for k, v in _LANG_DIRS.items()}


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


def _discover_library_metadata() -> tuple[
    dict[str, str],
    dict[tuple[str, str], str],
]:
    """Scan tests/<language>/<library>/metadata.json for display_name and repo entries.

    Returns (display_names, native_repos) where:
    - display_names maps library directory slug → display name.
    - native_repos maps (language slug, library slug) → GitHub repo slug
      for native ecosystem tests.
    Libraries that appear under multiple languages only need one metadata.json
    with a display_name (the first one found wins, they should all agree).
    Falls back to the slug itself when no metadata is found.
    """
    names: dict[str, str] = {}
    repos: dict[tuple[str, str], str] = {}
    if not TESTS_DIR.is_dir():
        return names, repos
    for lang_dir in sorted(TESTS_DIR.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name not in _LANG_DIRS:
            continue
        for lib_dir in sorted(lang_dir.iterdir()):
            if not lib_dir.is_dir():
                continue
            slug = lib_dir.name
            meta = lib_dir / "metadata.json"
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                if slug not in names and "display_name" in data:
                    names[slug] = data["display_name"]
                if "repo" in data:
                    repos[(lang_dir.name, slug)] = data["repo"]
            except (OSError, json.JSONDecodeError):
                pass
    return names, repos


_LIBRARY_DISPLAY_NAMES, NATIVE_REPOS = _discover_library_metadata()


def _library_display_name(slug: str) -> str:
    """Return the human-readable display name for a library slug."""
    return _LIBRARY_DISPLAY_NAMES.get(slug, slug)

# Deprecated attributes: shown with yellow background when present.
DEPRECATED_ATTRS = {
    "gen_ai.system",
}

# ── Data structures ──────────────────────────────────────────────────


def parse_results():
    """Parse all Weaver output directories under tests/.

    Layout: tests/<lang>/<lib>/results/<eco>/
    """
    results = {}

    if not TESTS_DIR.exists():
        print(f"Tests directory not found: {TESTS_DIR}", file=sys.stderr)
        return results

    for eco_dir in sorted(d for d in TESTS_DIR.glob("*/*/results/*") if d.is_dir()):
        eco = eco_dir.name
        lib = eco_dir.parent.parent.name
        lang = eco_dir.parent.parent.parent.name
        test_name = f"{lang}-{lib}-{eco}"
        r = parse_result_dir(eco_dir, test_name)
        if r and _has_detail_content(r):
            results[test_name] = r

    return results


def _has_detail_content(result: TestResult) -> bool:
    """Return whether a parsed result contains any renderable detail content."""
    return any([
        result.statistics is not None,
        result.has_data,
        bool(result.violation_messages),
        bool(result.seen_attrs),
        bool(result.seen_non_registry_attrs),
        bool(result.seen_events),
        bool(result.detected_span_types),
        bool(result.detected_events),
        bool(result.detected_metrics),
    ])


# ── HTML generation ──────────────────────────────────────────────────


def _compute_rowspans(rows: list[dict]) -> None:
    """Compute lib_rowspan / lang_rowspan for Library → Language hierarchy."""
    for r in rows:
        r["lang_rowspan"] = 0
        r["lib_rowspan"] = 0
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


def _prepare_details(
    results: dict[str, TestResult],
    test_data_entries: list[dict],
) -> list[dict]:
    """Prepare detailed result data for the template.

    Every known test (from committed data-*.json files) gets an anchor.
    Entries without local Weaver output show "Results not yet available."
    """
    result_by_id: dict[str, TestResult] = {
        _make_anchor_id(r.language, r.library, r.ecosystem): r
        for r in results.values()
    }

    _LANG_ORDER = {"python": 0, "js": 1, "java": 2, "c#": 3}
    sorted_entries = sorted(test_data_entries, key=lambda e: (
        e.get("library", "").lower(),
        _LANG_ORDER.get(e.get("language", "").lower(), 99),
        e.get("ecosystem", "").lower(),
    ))

    # Also include any results that have no corresponding data file.
    seen_ids = {e["test_name"] for e in sorted_entries}
    extra_results = sorted(
        (r for r in results.values()
         if _make_anchor_id(r.language, r.library, r.ecosystem) not in seen_ids),
        key=lambda r: (
            _library_display_name(r.library).lower(),
            r.language.lower(),
            r.ecosystem.lower(),
        ),
    )

    details = []

    def _append(
        anchor_id: str,
        label: str,
        lang_slug: str,
        lib: str,
        eco: str,
        language: str,
        r: TestResult | None,
    ) -> None:
        instrumentation_version = extract_version_from_deps(lang_slug, lib, eco)
        if eco == "native":
            repo = NATIVE_REPOS.get((lang_slug, lib), "")
        else:
            repo = ECOSYSTEM_REPOS.get((eco, language), "")

        detail: dict = {
            "test_name": anchor_id,
            "label": label,
            "has_data": bool(r and r.has_data),
            "violation_count": r.violation_count if r else 0,
            "instrumentation_version": instrumentation_version,
            "repo": repo,
            "entity_summary": "",
            "span_sections": [],
            "non_registry_attrs": [],
            "events": [],
            "metrics": [],
            "violation_messages": [],
        }

        if r and r.has_data:
            # Entity summary
            entity_parts = []
            for etype in ("span", "log", "resource", "attribute"):
                count = r.entity_counts.get(etype, 0)
                if count > 0:
                    entity_parts.append(f"{count} {etype}{'s' if count != 1 else ''}")
            detail["entity_summary"] = ", ".join(entity_parts)

            # Span-type attribute checklists
            all_present = set(r.seen_attrs) | set(r.seen_non_registry_attrs)

            for st_key in SPAN_TYPE_ORDER:
                spec = SPAN_TYPE_SPECS[st_key]
                all_spec_attrs: set[str] = set()
                for level in ("required", "conditionally_required", "recommended"):
                    all_spec_attrs.update(spec.get(level, []))

                discriminators = spec.get("discriminator_attrs", set())
                if discriminators:
                    if not ((all_present & discriminators)
                            or st_key in r.detected_span_types):
                        continue
                elif not (all_present & all_spec_attrs):
                    continue

                type_present = r.per_type_attrs.get(st_key, all_present)
                groups = []
                for level, level_label in [("required", "Required"),
                                           ("conditionally_required", "Conditionally Required"),
                                           ("recommended", "Recommended")]:
                    expected = spec.get(level, [])
                    if not expected:
                        continue
                    attrs = []
                    for attr in expected:
                        if attr in type_present:
                            count = r.seen_attrs.get(attr, r.seen_non_registry_attrs.get(attr, 0))
                            attrs.append({"name": attr, "present": True, "count": count})
                        else:
                            attrs.append({"name": attr, "present": False, "count": 0})
                    groups.append({"label": level_label, "attrs": attrs})

                detail["span_sections"].append({"label": spec["label"], "groups": groups})

            # Non-registry attributes
            if r.seen_non_registry_attrs:
                detail["non_registry_attrs"] = [
                    {"name": a, "count": c}
                    for a, c in sorted(r.seen_non_registry_attrs.items())
                ]

            merged_events = _merge_signal_counts(r.seen_events, r.detected_events)
            if merged_events:
                detail["events"] = [
                    {"name": a, "count": c}
                    for a, c in sorted(merged_events.items())
                ]

            merged_metrics = _merge_signal_counts(r.seen_metrics, r.detected_metrics)
            if merged_metrics:
                detail["metrics"] = [
                    {"name": a, "count": c}
                    for a, c in sorted(merged_metrics.items())
                ]

        if r and r.violation_messages:
            detail["violation_messages"] = r.violation_messages

        details.append(detail)

    for entry in sorted_entries:
        lang = entry["_lang"]
        lib = entry["_lib"]
        eco = entry["_eco"]
        language = entry.get("language", _LANG_DIRS.get(lang, lang))
        lib_display = entry.get("library", _library_display_name(lib))
        eco_display = entry.get("ecosystem", ECOSYSTEM_DISPLAY.get(eco, eco))
        label = f"{lib_display} ({language}) \u2014 {eco_display}"
        r = result_by_id.get(entry["test_name"])
        _append(entry["test_name"], label, lang, lib, eco, language, r)

    for r in extra_results:
        lang_slug = _LANG_SLUG.get(r.language, r.language.lower())
        anchor_id = _make_anchor_id(r.language, r.library, r.ecosystem)
        lib_display = _library_display_name(r.library)
        eco_display = ECOSYSTEM_DISPLAY.get(r.ecosystem, r.ecosystem)
        label = f"{lib_display} ({r.language}) \u2014 {eco_display}"
        _append(anchor_id, label, lang_slug, r.library, r.ecosystem, r.language, r)

    return details


def _load_test_data_files() -> list[dict]:
    """Discover and load all data-*.json files from tests/.

    Returns a list of loaded data dicts (each augmented with the file path).
    """
    entries: list[dict] = []
    if not TESTS_DIR.is_dir():
        return entries
    for data_file in sorted(TESTS_DIR.glob("**/data-*.json")):
        try:
            data = json.loads(data_file.read_text(encoding="utf-8"))
            # Derive fields from file path: tests/<lang>/<lib>/data-<eco>.json
            eco = data_file.stem.removeprefix("data-")
            lib = data_file.parent.name
            lang = data_file.parent.parent.name
            data.setdefault("library", _library_display_name(lib))
            data.setdefault("language", _LANG_DIRS.get(lang, lang))
            data.setdefault("ecosystem", ECOSYSTEM_DISPLAY.get(eco, eco))
            data["test_name"] = _make_anchor_id(data["language"], lib, eco)
            data["_lang"] = lang
            data["_lib"] = lib
            data["_eco"] = eco
            entries.append(data)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {data_file}: {e}", file=sys.stderr)
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
        relevant = [e for e in test_data_entries if st_key in e.get("span_types", {})]
        if not relevant:
            continue

        # Build column definitions from the spec.
        columns = []
        col_defs: list[tuple[str, bool]] = []
        for level in ("required", "conditionally_required", "recommended"):
            attrs = spec.get(level, [])
            for i, attr in enumerate(attrs):
                is_group_start = i == 0
                header_text = attr
                columns.append({"header_text": header_text, "is_group_start": is_group_start})
                col_defs.append((attr, is_group_start))

        if not columns:
            continue

        # Sort by library, then language (custom order), then ecosystem.
        _LANG_ORDER = {"python": 0, "js": 1, "java": 2, "c#": 3}
        relevant.sort(key=lambda e: (
            e.get("library", "").lower(),
            _LANG_ORDER.get(e.get("language", "").lower(), 99),
            e.get("ecosystem", "").lower(),
        ))

        rows = []
        for entry in relevant:
            attr_statuses = entry["span_types"][st_key]
            cells = []
            for attr, is_group_start in col_defs:
                status = attr_statuses.get(attr, "absent")
                if status == "present":
                    if attr in DEPRECATED_ATTRS:
                        cls, symbol = "deprecated", "\u2713"
                    else:
                        cls, symbol = "present", "\u2713"
                else:
                    cls, symbol = "absent", ""
                group_cls = " group-start" if is_group_start else ""
                cells.append({"cls": f"{cls}{group_cls}", "symbol": symbol})

            rows.append({
                "test_name": entry.get("test_name", ""),
                "has_details": details_available,
                "lib_display": entry.get("library", ""),
                "language": entry.get("language", ""),
                "eco_display": entry.get("ecosystem", ""),
                "instrumentation_version": extract_version_from_deps(
                    entry["_lang"], entry["_lib"], entry["_eco"],
                ),
                "cells": cells,
            })

        _compute_rowspans(rows)
        heatmaps.append({"label": st_label, "columns": columns, "rows": rows})

    return heatmaps


def _merge_signal_counts(
    statistics_counts: dict[str, int],
    detected_counts: dict[str, int],
) -> dict[str, int]:
    """Merge statistic-derived and sample-derived signal counts."""
    merged = dict(statistics_counts)
    for name, count in detected_counts.items():
        merged[name] = max(merged.get(name, 0), count)
    return merged


def _prepare_signal_heatmap(
    test_data_entries: list[dict],
    signal_names: list[str],
    label: str,
    data_key: str,
    details_available: bool = True,
) -> dict | None:
    """Build an event or metric heatmap from committed data-*.json entries."""
    relevant = [entry for entry in test_data_entries if data_key in entry]

    if not relevant:
        return None

    columns = [
        {"header_text": signal_name, "is_group_start": index == 0}
        for index, signal_name in enumerate(signal_names)
    ]

    language_order = {"Python": 0, "JS": 1, "Java": 2, "C#": 3}
    relevant.sort(key=lambda entry: (
        entry.get("library", "").lower(),
        language_order.get(entry.get("language", ""), 99),
        entry.get("ecosystem", "").lower(),
    ))

    rows = []
    for entry in relevant:
        signal_statuses = entry.get(data_key, {})
        cells = []
        for index, signal_name in enumerate(signal_names):
            group_cls = " group-start" if index == 0 else ""
            if signal_statuses.get(signal_name) == "present":
                cells.append({"cls": f"present{group_cls}", "symbol": "\u2713"})
            else:
                cells.append({"cls": f"absent{group_cls}", "symbol": ""})

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
            "cells": cells,
        })

    _compute_rowspans(rows)
    return {"label": label, "columns": columns, "rows": rows}


def _has_result_directories() -> bool:
    """Return whether any local Weaver result directories exist."""
    return any(d.is_dir() for d in TESTS_DIR.glob("*/*/results/*"))


def generate_dashboard_html(details_available: bool) -> str:
    """Generate the dashboard HTML with span heatmap tables.

    All dashboard heatmaps come from committed data-*.json files.
    """
    import jinja2

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    test_data_entries = _load_test_data_files()
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

    css = (TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("dashboard.html")
    return template.render(
        css=css, now=now, heatmaps=heatmaps,
        details_available=details_available,
        event_heatmap=event_heatmap, metric_heatmap=metric_heatmap,
    )


def generate_details_html(
    results: dict[str, TestResult],
    test_data_entries: list[dict],
) -> str:
    """Generate the details HTML from Weaver results."""
    import jinja2

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    details = _prepare_details(results, test_data_entries)

    css = (TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("details.html")
    return template.render(css=css, now=now, details=details)


# ── Main ─────────────────────────────────────────────────────────────


def main():
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
    dashboard_html = generate_dashboard_html(details_available)
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
