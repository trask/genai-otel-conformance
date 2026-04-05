"""Span classification heuristics for GenAI OTel conformance results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class SpanClassification:
    detected_types: set[str] = field(default_factory=set)
    per_type_attrs: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class DetectedSignals:
    events: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, int] = field(default_factory=dict)
    event_attrs: dict[str, set[str]] = field(default_factory=dict)
    event_any_attrs: dict[str, set[str]] = field(default_factory=dict)
    metric_attrs: dict[str, set[str]] = field(default_factory=dict)
    metric_any_attrs: dict[str, set[str]] = field(default_factory=dict)


def _has_any_attr(attrs: dict[str, object], *names: str) -> bool:
    return any(attrs.get(name) is not None for name in names)


def _has_all_attrs(attrs: dict[str, object], *names: str) -> bool:
    return all(attrs.get(name) is not None for name in names)


def _has_attr_prefix(attrs: dict[str, object], prefix: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for name in attrs)


class SpanInfo(NamedTuple):
    """Pre-extracted span fields passed to each classifier."""
    name_lower: str
    op_name: str
    oi_kind: str
    llm_type: str
    attrs: dict[str, object]


def _is_embeddings_span(ctx: SpanInfo) -> bool:
    return (
        "embed" in ctx.name_lower
        or _has_any_attr(ctx.attrs, "embedding.model_name")
        or ctx.oi_kind == "EMBEDDING"
        or ctx.llm_type in ("embedding", "embeddings")
        or ctx.op_name in ("embedding", "embeddings")
    )


def _is_inference_span(ctx: SpanInfo) -> bool:
    return (
        ctx.op_name == "chat"
        or ctx.oi_kind == "LLM"
        or ctx.llm_type in ("chat", "completion")
        or ctx.op_name == "generate_content"
        or _has_all_attrs(ctx.attrs, "gen_ai.usage.output_tokens", "gen_ai.response.finish_reasons")
        or _has_all_attrs(ctx.attrs, "llm.response.model", "llm.usage.completion_tokens")
    )


def _is_create_agent_span(ctx: SpanInfo) -> bool:
    return ctx.op_name == "create_agent"


def _is_invoke_agent_like(ctx: SpanInfo) -> bool:
    return (
        ctx.oi_kind == "AGENT"
        or ctx.op_name == "invoke_agent"
        or (
            _has_any_attr(ctx.attrs, "gen_ai.agent.name", "gen_ai.agent.id")
            and ctx.op_name != "create_agent"
        )
        or _has_any_attr(ctx.attrs, "crewai.agent.id", "crewai.agent.role")
        or (
            str(ctx.attrs.get("rpc.service", "")).lower() == "bedrockagentruntime"
            and str(ctx.attrs.get("rpc.method", "")).lower() == "invokeagent"
        )
        or ("agentsclient" in ctx.name_lower and ("run" in ctx.name_lower or "process" in ctx.name_lower))
        or ("threads" in ctx.name_lower and "run" in ctx.name_lower and "thread.run" not in ctx.name_lower)
    )


def _is_remote_agent(ctx: SpanInfo) -> bool:
    return (
        _has_any_attr(ctx.attrs, "server.address", "server.port")
        or (
            str(ctx.attrs.get("rpc.service", "")).lower() == "bedrockagentruntime"
            and str(ctx.attrs.get("rpc.method", "")).lower() == "invokeagent"
        )
        or ("agentsclient" in ctx.name_lower and ("run" in ctx.name_lower or "process" in ctx.name_lower))
        or ("threads" in ctx.name_lower and "run" in ctx.name_lower and "thread.run" not in ctx.name_lower)
    )


def _is_invoke_agent_span(ctx: SpanInfo) -> bool:
    return _is_invoke_agent_like(ctx) and _is_remote_agent(ctx)


def _is_invoke_agent_internal_span(ctx: SpanInfo) -> bool:
    return _is_invoke_agent_like(ctx) and not _is_remote_agent(ctx)


def _is_execute_tool_span(ctx: SpanInfo) -> bool:
    return (
        ctx.op_name == "execute_tool"
        or _has_any_attr(ctx.attrs, "gen_ai.tool.name", "gen_ai.tool.call.id")
        or _has_any_attr(ctx.attrs, "tool.name", "tool.id")
        or (ctx.oi_kind == "TOOL" and "tool" in ctx.name_lower)
    )


def _is_invoke_workflow_span(ctx: SpanInfo) -> bool:
    return (
        ctx.op_name == "invoke_workflow"
        or _has_any_attr(ctx.attrs, "traceloop.workflow.name")
        or ctx.name_lower == "crewai.workflow"
        or _has_any_attr(ctx.attrs, "crewai.crew.id")
    )


def _is_retrieval_span(ctx: SpanInfo) -> bool:
    return (
        ctx.op_name == "retrieval"
        or ctx.oi_kind == "RETRIEVER"
        or _has_any_attr(ctx.attrs, "gen_ai.data_source.id")
    )


_SPAN_TYPE_CLASSIFIERS: list[tuple[str, Callable[[SpanInfo], bool]]] = [
    ("embeddings", _is_embeddings_span),
    ("inference", _is_inference_span),
    ("create_agent", _is_create_agent_span),
    ("invoke_agent", _is_invoke_agent_span),
    ("invoke_agent_internal", _is_invoke_agent_internal_span),
    ("execute_tool", _is_execute_tool_span),
    ("invoke_workflow", _is_invoke_workflow_span),
    ("retrieval", _is_retrieval_span),
]


def _classify_span(span_name: str, span_attrs: dict[str, object]) -> set[str]:
    """Classify a span into span types using heuristics on individual span data."""
    ctx = SpanInfo(
        name_lower=span_name.lower(),
        op_name=_infer_operation_name(span_name, span_attrs),
        oi_kind=str(span_attrs.get("openinference.span.kind", "")).upper(),
        llm_type=str(span_attrs.get("llm.request.type", "")).lower(),
        attrs=span_attrs,
    )

    return {
        span_type
        for span_type, predicate in _SPAN_TYPE_CLASSIFIERS
        if predicate(ctx)
    }


def _infer_operation_name(span_name: str, attrs: dict[str, object]) -> str:
    """Infer span operation name for classification without rewriting raw attrs."""
    op_name = str(attrs.get("gen_ai.operation.name", "")).lower()
    if op_name:
        return op_name

    name_lower = span_name.lower()
    oi_kind = str(attrs.get("openinference.span.kind", "")).upper()

    if _has_attr_prefix(attrs, "llm.input_messages") or "chat" in name_lower:
        return "chat"
    if "embedding" in name_lower or attrs.get("embedding.model_name") is not None:
        return "embeddings"
    if oi_kind == "AGENT":
        return "invoke_agent"
    if oi_kind == "RETRIEVER":
        return "retrieval"
    if oi_kind == "TOOL" and (
        attrs.get("tool.name") is not None or attrs.get("tool.id") is not None or "tool" in name_lower
    ):
        return "execute_tool"
    return ""


def _span_attributes(span: dict[str, object]) -> dict[str, object]:
    return _span_attributes_filtered(span)


def _span_attributes_filtered(
    span: dict[str, object],
    include_attr: Callable[[dict[str, object]], bool] | None = None,
) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for attr in span.get("attributes", []):
        if not isinstance(attr, dict):
            continue
        if include_attr is not None and not include_attr(attr):
            continue
        attrs[attr.get("name", "")] = attr.get("value")
    return attrs


def _span_attribute_names(
    span: dict[str, object],
    include_attr: Callable[[dict[str, object]], bool] | None = None,
) -> set[str]:
    names: set[str] = set()
    for attr in span.get("attributes", []):
        if not isinstance(attr, dict):
            continue
        name = attr.get("name")
        if not isinstance(name, str) or not name:
            continue
        if include_attr is not None and not include_attr(attr):
            continue
        names.add(name)
    return names


def _metric_attribute_names(
    metric: dict[str, object],
    include_attr: Callable[[dict[str, object]], bool] | None = None,
) -> set[str]:
    names: set[str] = set()
    for dp in metric.get("data_points", []):
        if not isinstance(dp, dict):
            continue
        for attr in dp.get("attributes", []):
            if not isinstance(attr, dict):
                continue
            name = attr.get("name")
            if not isinstance(name, str) or not name:
                continue
            if include_attr is not None and not include_attr(attr):
                continue
            names.add(name)
    return names


def summarize_samples(
    all_objects: list[dict],
    include_attr: Callable[[dict[str, object]], bool] | None = None,
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
                attrs = _span_attributes_filtered(span, include_attr)
                classified = _classify_span(span.get("name", ""), attrs)
                spans.detected_types.update(classified)
                attr_names = _span_attribute_names(span, include_attr)
                for span_type in classified:
                    spans.per_type_attrs.setdefault(span_type, set()).update(attr_names)

            log = sample.get("log")
            if log:
                event_name = log.get("event_name", "")
                if event_name.startswith("gen_ai."):
                    signals.events[event_name] = signals.events.get(event_name, 0) + 1
                    attr_names = _span_attribute_names(log, include_attr)
                    signals.event_attrs.setdefault(event_name, set()).update(attr_names)
                    signals.event_any_attrs.setdefault(event_name, set()).update(attr_names)

            metric = sample.get("metric")
            if metric:
                metric_name = metric.get("name", "")
                if metric_name.startswith("gen_ai."):
                    signals.metrics[metric_name] = signals.metrics.get(metric_name, 0) + 1
                    attr_names = _metric_attribute_names(metric, include_attr)
                    if metric_name not in signals.metric_attrs:
                        signals.metric_attrs[metric_name] = set(attr_names)
                    else:
                        signals.metric_attrs[metric_name].intersection_update(attr_names)
                    signals.metric_any_attrs.setdefault(metric_name, set()).update(attr_names)

    return spans, signals
