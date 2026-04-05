"""Semantic convention span type specs, event types, and metric types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RequirementLevel(StrEnum):
    REQUIRED = "required"
    CONDITIONALLY_REQUIRED = "conditionally_required"
    RECOMMENDED = "recommended"
    OPT_IN = "opt_in"


@dataclass(frozen=True)
class SignalTypeSpec:
    label: str
    required: tuple[str, ...]
    conditionally_required: tuple[str, ...]
    recommended: tuple[str, ...]
    opt_in: tuple[str, ...]
    discriminator_attrs: frozenset[str] = frozenset()

    def attrs_for_requirement_level(self, level: RequirementLevel) -> tuple[str, ...]:
        if level is RequirementLevel.REQUIRED:
            return self.required
        if level is RequirementLevel.CONDITIONALLY_REQUIRED:
            return self.conditionally_required
        if level is RequirementLevel.RECOMMENDED:
            return self.recommended
        if level is RequirementLevel.OPT_IN:
            return self.opt_in
        raise KeyError(f"Unknown requirement level: {level}")


_COMMON_REQUIRED = ["gen_ai.operation.name"]
_PROVIDER_REQUIRED = ["gen_ai.provider.name"]
# This deprecated attr is rendered next to its canonical attr in the same
# dashboard group, but it is not a semconv requirement.
DISPLAY_DEPRECATED_ATTRS = {
    "gen_ai.provider.name": "gen_ai.system",
}
_COMMON_COND_REQUIRED = ["error.type"]
_CLIENT_COND_REQUIRED = ["gen_ai.request.model", "server.port"]
_CLIENT_RECOMMENDED = ["server.address"]
_INFERENCE_COND_REQUIRED = [
    "gen_ai.conversation.id",
    "gen_ai.output.type",
    "gen_ai.request.choice.count",
    "gen_ai.request.seed",
]
_INFERENCE_RECOMMENDED = [
    "gen_ai.request.frequency_penalty",
    "gen_ai.request.max_tokens",
    "gen_ai.request.presence_penalty",
    "gen_ai.request.stop_sequences",
    "gen_ai.request.temperature",
    "gen_ai.request.top_p",
    "gen_ai.response.finish_reasons",
    "gen_ai.response.id",
    "gen_ai.response.model",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.reasoning.output_tokens",
]
_INFERENCE_OPT_IN = [
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
]

_INVOKE_AGENT_COND_REQUIRED = [
    "gen_ai.conversation.id",
    "gen_ai.output.type",
    "gen_ai.request.choice.count",
    "gen_ai.request.seed",
    "gen_ai.agent.description",
    "gen_ai.agent.id",
    "gen_ai.agent.name",
    "gen_ai.agent.version",
    "gen_ai.data_source.id",
]
_INVOKE_AGENT_RECOMMENDED = [
    "gen_ai.request.frequency_penalty",
    "gen_ai.request.max_tokens",
    "gen_ai.request.presence_penalty",
    "gen_ai.request.stop_sequences",
    "gen_ai.request.temperature",
    "gen_ai.request.top_p",
    "gen_ai.response.finish_reasons",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
]

SPAN_TYPE_SPECS: dict[str, SignalTypeSpec] = {
    "inference": SignalTypeSpec(
        label="Inference",
        discriminator_attrs=frozenset({
            "gen_ai.response.finish_reasons",
            "gen_ai.response.id",
            "gen_ai.usage.output_tokens",
            "gen_ai.request.max_tokens",
            "gen_ai.request.temperature",
            "gen_ai.output.type",
            "gen_ai.usage.input_tokens",
        }),
        required=tuple(_COMMON_REQUIRED + _PROVIDER_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INFERENCE_COND_REQUIRED),
        recommended=tuple(_INFERENCE_RECOMMENDED + ["gen_ai.request.top_k"] + _CLIENT_RECOMMENDED),
        opt_in=tuple(_INFERENCE_OPT_IN),
    ),
    "embeddings": SignalTypeSpec(
        label="Embeddings",
        discriminator_attrs=frozenset({
            "gen_ai.embeddings.dimension.count",
            "gen_ai.request.encoding_formats",
        }),
        required=tuple(_COMMON_REQUIRED + _PROVIDER_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED),
        recommended=tuple([
            "gen_ai.embeddings.dimension.count",
            "gen_ai.request.encoding_formats",
            "gen_ai.response.model",
            "gen_ai.usage.input_tokens",
        ] + _CLIENT_RECOMMENDED),
        opt_in=(),
    ),
    "retrieval": SignalTypeSpec(
        label="Retrieval",
        discriminator_attrs=frozenset({"gen_ai.data_source.id"}),
        required=tuple(_COMMON_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED + [
            "gen_ai.data_source.id",
            "gen_ai.provider.name",
        ] + _CLIENT_COND_REQUIRED),
        recommended=tuple(["gen_ai.request.top_k"] + _CLIENT_RECOMMENDED),
        opt_in=(
            "gen_ai.retrieval.documents",
            "gen_ai.retrieval.query.text",
        ),
    ),
    "execute_tool": SignalTypeSpec(
        label="Execute Tool",
        discriminator_attrs=frozenset({
            "gen_ai.tool.call.id",
            "gen_ai.tool.name",
            "gen_ai.tool.type",
        }),
        required=tuple(_COMMON_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED),
        recommended=(
            "gen_ai.tool.call.id",
            "gen_ai.tool.description",
            "gen_ai.tool.name",
            "gen_ai.tool.type",
        ),
        opt_in=(
            "gen_ai.tool.call.arguments",
            "gen_ai.tool.call.result",
        ),
    ),
    "create_agent": SignalTypeSpec(
        label="Create Agent",
        discriminator_attrs=frozenset({"gen_ai.agent.id", "gen_ai.agent.name"}),
        required=tuple(_COMMON_REQUIRED + _PROVIDER_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.agent.version",
        ]),
        recommended=tuple(_CLIENT_RECOMMENDED),
        opt_in=("gen_ai.system_instructions",),
    ),
    "invoke_agent": SignalTypeSpec(
        label="Invoke Agent Client",
        discriminator_attrs=frozenset({"gen_ai.agent.id", "gen_ai.agent.name"}),
        required=tuple(_COMMON_REQUIRED + _PROVIDER_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INVOKE_AGENT_COND_REQUIRED),
        recommended=tuple(_INVOKE_AGENT_RECOMMENDED + _CLIENT_RECOMMENDED),
        opt_in=tuple(_INFERENCE_OPT_IN),
    ),
    "invoke_agent_internal": SignalTypeSpec(
        label="Invoke Agent Internal",
        discriminator_attrs=frozenset({"gen_ai.agent.id", "gen_ai.agent.name"}),
        required=tuple(_COMMON_REQUIRED + _PROVIDER_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED + ["gen_ai.request.model"] + _INVOKE_AGENT_COND_REQUIRED),
        recommended=tuple(_INVOKE_AGENT_RECOMMENDED),
        opt_in=tuple(_INFERENCE_OPT_IN),
    ),
    "invoke_workflow": SignalTypeSpec(
        label="Invoke Workflow",
        discriminator_attrs=frozenset({"gen_ai.workflow.name"}),
        required=tuple(_COMMON_REQUIRED),
        conditionally_required=tuple(_COMMON_COND_REQUIRED + ["gen_ai.workflow.name"]),
        recommended=(),
        opt_in=(
            "gen_ai.input.messages",
            "gen_ai.output.messages",
        ),
    ),
}

SPAN_TYPE_ORDER = [
    "create_agent",
    "invoke_agent",
    "invoke_agent_internal",
    "invoke_workflow",
    "inference",
    "embeddings",
    "retrieval",
    "execute_tool",
]

EVENT_TYPE_SPECS: dict[str, SignalTypeSpec] = {
    "gen_ai.client.inference.operation.details": SignalTypeSpec(
        label="Inference Operation Details Event",
        required=("gen_ai.operation.name",),
        conditionally_required=(
            "error.type",
            "gen_ai.request.model",
            "server.port",
        ),
        recommended=(
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.response.finish_reasons",
            "gen_ai.response.id",
            "gen_ai.response.model",
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "server.address",
        ),
        opt_in=(),
    ),
    "gen_ai.evaluation.result": SignalTypeSpec(
        label="Evaluation Result Event",
        required=("gen_ai.evaluation.name",),
        conditionally_required=(),
        recommended=(
            "gen_ai.evaluation.explanation",
            "gen_ai.evaluation.score.label",
            "gen_ai.evaluation.score.value",
            "gen_ai.response.id",
        ),
        opt_in=(),
    ),
}

METRIC_TYPE_SPECS: dict[str, SignalTypeSpec] = {
    "gen_ai.client.operation.duration": SignalTypeSpec(
        label="Client Operation Duration Metric",
        required=("gen_ai.operation.name",),
        conditionally_required=(
            "error.type",
            "gen_ai.request.model",
            "server.port",
        ),
        recommended=(
            "gen_ai.response.model",
            "server.address",
        ),
        opt_in=(),
    ),
    "gen_ai.client.token.usage": SignalTypeSpec(
        label="Client Token Usage Metric",
        required=(
            "gen_ai.operation.name",
            "gen_ai.token.type",
        ),
        conditionally_required=(
            "gen_ai.request.model",
            "server.port",
        ),
        recommended=(
            "gen_ai.response.model",
            "server.address",
        ),
        opt_in=(),
    ),
}

GENAI_EVENT_TYPES: dict[str, str] = {k: v.label for k, v in EVENT_TYPE_SPECS.items()}
GENAI_METRIC_TYPES: dict[str, str] = {k: v.label for k, v in METRIC_TYPE_SPECS.items()}
