"""View dataclasses for Jinja template rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

from genai_otel_conformance.statuses import HeatmapColumn, HeatmapGroup


@dataclass(frozen=True)
class StatusCell:
    cls: str
    symbol: str


@dataclass
class HeatmapRow:
    test_name: str
    has_details: bool
    lib_display: str
    language: str
    eco_display: str
    instrumentation_version: str
    cells: list[StatusCell]
    lib_rowspan: int = 0
    lang_rowspan: int = 0


@dataclass(frozen=True)
class HeatmapView:
    label: str
    anchor_id: str
    columns: list[HeatmapColumn]
    column_groups: list[HeatmapGroup]
    rows: list[HeatmapRow]


@dataclass(frozen=True)
class DetailAttributeView:
    name: str
    present: bool
    count: int


@dataclass(frozen=True)
class DetailGroupView:
    label: str
    attrs: list[DetailAttributeView]


@dataclass(frozen=True)
class SpanSectionView:
    label: str
    groups: list[DetailGroupView]


@dataclass(frozen=True)
class CountItemView:
    name: str
    count: int


@dataclass(frozen=True)
class DetailView:
    test_name: str
    label: str
    has_local_run: bool
    has_data: bool
    has_empty_run: bool
    violation_count: int
    instrumentation_version: str
    repo: str
    entity_summary: str
    span_sections: list[SpanSectionView] = field(default_factory=list)
    non_registry_attrs: list[CountItemView] = field(default_factory=list)
    metrics: list[CountItemView] = field(default_factory=list)
    events: list[CountItemView] = field(default_factory=list)
    violation_messages: list[str] = field(default_factory=list)
