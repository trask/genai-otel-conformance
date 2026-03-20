#!/usr/bin/env python3
"""Generate a static HTML conformance dashboard from Weaver live-check results.

Usage:
    python generate_dashboard.py [--output-dir DIR]

Reads Weaver JSON output from tests/<lang>/<lib>/results/<eco>/ directories
and produces a static HTML dashboard at <output-dir>/index.html.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from run_test import (
    _LANG_DIRS,
    ECOSYSTEM_DISPLAY,
    ECOSYSTEM_REPOS,
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

    Args:
        language: Display language name (e.g. "Python", "JS").
        library: Library slug (e.g. "openai").
        ecosystem: Ecosystem slug (e.g. "otelcontrib").
    """
    lang_slug = _LANG_SLUG.get(language, language.lower())
    return f"{library}-{lang_slug}-{ecosystem}"


def _discover_library_metadata() -> tuple[dict[str, str], dict[str, str]]:
    """Scan tests/<language>/<library>/metadata.json for display_name and repo entries.

    Returns (display_names, native_repos) where:
    - display_names maps library directory slug → display name.
    - native_repos maps library directory slug → GitHub repo slug (for native ecosystem).
    Libraries that appear under multiple languages only need one metadata.json
    with a display_name (the first one found wins, they should all agree).
    Falls back to the slug itself when no metadata is found.
    """
    names: dict[str, str] = {}
    repos: dict[str, str] = {}
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
                if slug not in repos and "repo" in data:
                    repos[slug] = data["repo"]
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

# ── GenAI event types (from gen-ai-events.md) ────────────────────────
# Events are emitted as OTel log records with an event_name field.

GENAI_EVENT_TYPES = [
    "gen_ai.system.message",
    "gen_ai.user.message",
    "gen_ai.assistant.message",
    "gen_ai.tool.message",
    "gen_ai.choice",
]

# ── GenAI metric types (from gen-ai-metrics.md) ─────────────────────

GENAI_METRIC_TYPES = [
    "gen_ai.client.token.usage",
    "gen_ai.client.operation.duration",
]


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class HeatmapRow:
    test_name: str
    lib_display: str
    eco_display: str
    present_attrs: set[str]
    library: str
    language: str
    repo: str
    instrumentation_version: str


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
        if r:
            results[test_name] = r

    return results


# ── HTML generation ──────────────────────────────────────────────────


def _build_heatmap_rows(results: dict[str, TestResult]) -> list[HeatmapRow]:
    """Build one HeatmapRow per test that has gen_ai.* attribute data
    or spans detected via heuristics (e.g. non-conforming embeddings)."""
    rows = []
    for test_name, r in results.items():
        if not r.has_data:
            continue
        all_present = set(r.seen_attrs) | set(r.seen_non_registry_attrs)
        has_genai = any(a.startswith("gen_ai.") for a in all_present)
        if not has_genai and not r.detected_span_types:
            continue

        lib_display = _library_display_name(r.library)
        eco_display = ECOSYSTEM_DISPLAY.get(r.ecosystem, r.ecosystem)
        if r.ecosystem == "native":
            repo = NATIVE_REPOS.get(r.library, r.ecosystem)
        else:
            repo = ECOSYSTEM_REPOS.get(
                (r.ecosystem, r.language),
                ECOSYSTEM_DISPLAY.get(r.ecosystem, r.ecosystem),
            )

        rows.append(HeatmapRow(
            test_name=test_name,
            lib_display=lib_display,
            eco_display=eco_display,
            present_attrs=all_present,
            library=r.library,
            language=r.language,
            repo=repo,
            instrumentation_version=extract_version_from_deps(
                r.language, r.library, r.ecosystem,
            ),
        ))

    rows.sort(key=lambda x: (x.language.lower(), x.lib_display.lower(), x.eco_display.lower()))
    return rows


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


def _prepare_simple_heatmap(
    label: str,
    type_names: list[str],
    heatmap_rows: list[HeatmapRow],
    results: dict[str, TestResult],
    detected_field: str,
) -> dict | None:
    """Prepare a simple heatmap (events or metrics) where columns are type names.

    detected_field is the TestResult attribute name holding a set[str] of
    detected type names (e.g. "detected_events" or "detected_metrics").
    """
    # Filter to rows that have at least one detected type
    relevant = [
        r for r in heatmap_rows
        if getattr(results[r.test_name], detected_field, set()) & set(type_names)
    ]
    if not relevant:
        return None

    # Discover additional types seen in the data but not in the standard list
    all_detected: set[str] = set()
    for r in relevant:
        all_detected |= getattr(results[r.test_name], detected_field, set())
    extra_types = sorted(all_detected - set(type_names))
    all_types = type_names + extra_types

    columns = [{"header_text": t, "is_group_start": False} for t in all_types]

    rows = []
    for row in relevant:
        detected = getattr(results[row.test_name], detected_field, set())
        cells = []
        for t in all_types:
            present = t in detected
            cls = "present" if present else "absent"
            symbol = "\u2713" if present else ""
            cells.append({"cls": cls, "symbol": symbol})

        short_version = ""
        if row.instrumentation_version:
            parts = row.instrumentation_version.rsplit(" ", 1)
            short_version = parts[-1] if len(parts) == 2 else row.instrumentation_version

        rows.append({
            "test_name": row.test_name,
            "lib_display": row.lib_display,
            "language": row.language,
            "eco_display": row.eco_display,
            "instrumentation_version": short_version,
            "cells": cells,
        })

    _compute_rowspans(rows)
    return {"label": label, "columns": columns, "rows": rows}


def _prepare_details(results: dict[str, TestResult]) -> list[dict]:
    """Prepare detailed result data for the template."""
    details = []

    for test_name in sorted(results):
        r = results[test_name]
        lib_display = _library_display_name(r.library)
        eco_display = ECOSYSTEM_DISPLAY.get(r.ecosystem, r.ecosystem)
        label = f"{lib_display} ({r.language}) \u2014 {eco_display}"

        lang_slug = _LANG_SLUG.get(r.language, r.language.lower())
        instrumentation_version = extract_version_from_deps(
            lang_slug, r.library, r.ecosystem,
        )

        if r.ecosystem == "native":
            repo = NATIVE_REPOS.get(r.library, "")
        else:
            repo = ECOSYSTEM_REPOS.get((r.ecosystem, r.language), "")

        detail: dict = {
            "test_name": _make_anchor_id(r.language, r.library, r.ecosystem),
            "label": label,
            "has_data": r.has_data,
            "violation_count": r.violation_count,
            "instrumentation_version": instrumentation_version,
            "repo": repo,
            "entity_summary": "",
            "span_sections": [],
            "non_registry_attrs": [],
            "events": [],
            "violation_messages": [],
        }

        if r.has_data:
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

                groups = []
                for level, level_label in [("required", "Required"),
                                           ("conditionally_required", "Conditionally Required"),
                                           ("recommended", "Recommended")]:
                    expected = spec.get(level, [])
                    if not expected:
                        continue
                    attrs = []
                    for attr in expected:
                        if attr in all_present:
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

            # Events
            if r.seen_events:
                detail["events"] = [
                    {"name": a, "count": c}
                    for a, c in sorted(r.seen_events.items())
                ]

        # Violation messages
        if r.violation_messages:
            detail["violation_messages"] = r.violation_messages

        details.append(detail)

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
            data["test_name"] = f"{lib}-{lang}-{eco}"
            data.setdefault("library", _library_display_name(lib))
            data.setdefault("language", _LANG_DIRS.get(lang, lang))
            data.setdefault("ecosystem", ECOSYSTEM_DISPLAY.get(eco, eco))
            data["_lang"] = lang
            data["_lib"] = lib
            data["_eco"] = eco
            entries.append(data)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {data_file}: {e}", file=sys.stderr)
    return entries


def _prepare_heatmaps_from_data(test_data_entries: list[dict]) -> list[dict]:
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


def generate_dashboard_html() -> str:
    """Generate the dashboard HTML with span heatmap tables.

    Built purely from committed data-*.json files — no Weaver results needed.
    """
    import jinja2

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    test_data_entries = _load_test_data_files()
    heatmaps = _prepare_heatmaps_from_data(test_data_entries)

    # Events and metrics are not yet in data files, pass empty for now.
    event_heatmap = None
    metric_heatmap = None

    css = (TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("dashboard.html")
    return template.render(
        css=css, now=now, heatmaps=heatmaps,
        event_heatmap=event_heatmap, metric_heatmap=metric_heatmap,
    )


def generate_details_html(results: dict[str, TestResult]) -> str:
    """Generate the details HTML from Weaver results."""
    import jinja2

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    details = _prepare_details(results)

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

    # Dashboard (index.html) is always generated from data files only.
    dashboard_html = generate_dashboard_html()
    (out / "index.html").write_text(dashboard_html, encoding="utf-8")
    print(f"Dashboard written to {out / 'index.html'}")

    # Details page from Weaver results.
    results = parse_results()
    if not results:
        print("No results found. Skipping details.", file=sys.stderr)
    else:
        details_html = generate_details_html(results)
        (out / "details.html").write_text(details_html, encoding="utf-8")
        print(f"Details written to {out / 'details.html'}")
        print(f"  Tests: {len(results)}")


if __name__ == "__main__":
    main()
