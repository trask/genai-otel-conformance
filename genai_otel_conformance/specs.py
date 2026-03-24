"""Semantic convention span type specs, event types, and metric types."""

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
]
_INFERENCE_OPT_IN = [
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
]

SPAN_TYPE_SPECS: dict[str, dict] = {
    "inference": {
        "label": "Inference",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.response.finish_reasons", "gen_ai.response.id",
            "gen_ai.usage.output_tokens", "gen_ai.request.max_tokens",
            "gen_ai.request.temperature", "gen_ai.output.type",
            "gen_ai.usage.input_tokens",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INFERENCE_COND_REQUIRED,
        "recommended": _INFERENCE_RECOMMENDED + ["gen_ai.request.top_k"] + _CLIENT_RECOMMENDED,
        "opt_in": _INFERENCE_OPT_IN,
    },
    "embeddings": {
        "label": "Embeddings",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.embeddings.dimension.count", "gen_ai.request.encoding_formats",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED,
        "recommended": [
            "gen_ai.embeddings.dimension.count",
            "gen_ai.request.encoding_formats",
            "gen_ai.response.model",
            "gen_ai.usage.input_tokens",
        ] + _CLIENT_RECOMMENDED,
        "opt_in": [],
    },
    "retrieval": {
        "label": "Retrieval",
        "expected_kind": "client",
        "discriminator_attrs": {"gen_ai.data_source.id"},
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + [
            "gen_ai.data_source.id",
            "gen_ai.provider.name",
        ] + _CLIENT_COND_REQUIRED,
        "recommended": ["gen_ai.request.top_k"] + _CLIENT_RECOMMENDED,
        "opt_in": [
            "gen_ai.retrieval.documents",
            "gen_ai.retrieval.query.text",
        ],
    },
    "execute_tool": {
        "label": "Execute Tool",
        "expected_kind": "internal",
        "discriminator_attrs": {
            "gen_ai.tool.call.id", "gen_ai.tool.name", "gen_ai.tool.type",
        },
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED,
        "recommended": [
            "gen_ai.tool.call.id",
            "gen_ai.tool.description",
            "gen_ai.tool.name",
            "gen_ai.tool.type",
        ],
        "opt_in": [
            "gen_ai.tool.call.arguments",
            "gen_ai.tool.call.result",
        ],
    },
    "create_agent": {
        "label": "Create Agent",
        "expected_kind": "client",
        "discriminator_attrs": {"gen_ai.agent.id", "gen_ai.agent.name"},
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.agent.version",
        ],
        "recommended": _CLIENT_RECOMMENDED,
        "opt_in": ["gen_ai.system_instructions"],
    },
    "invoke_agent": {
        "label": "Invoke Agent",
        "expected_kind": "client",
        "discriminator_attrs": {"gen_ai.agent.id", "gen_ai.agent.name"},
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INFERENCE_COND_REQUIRED + [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.agent.version",
            "gen_ai.data_source.id",
        ],
        "recommended": _INFERENCE_RECOMMENDED + _CLIENT_RECOMMENDED,
        "opt_in": _INFERENCE_OPT_IN,
    },
    "invoke_workflow": {
        "label": "Invoke Workflow",
        "expected_kind": "internal",
        "discriminator_attrs": {"gen_ai.workflow.name"},
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + ["gen_ai.workflow.name"],
        "recommended": [],
        "opt_in": [
            "gen_ai.input.messages",
            "gen_ai.output.messages",
        ],
    },
}

SPAN_TYPE_ORDER = [
    "create_agent",
    "invoke_agent",
    "invoke_workflow",
    "inference",
    "embeddings",
    "retrieval",
    "execute_tool",
]

GENAI_EVENT_TYPES = [
    "gen_ai.system.message",
    "gen_ai.user.message",
    "gen_ai.assistant.message",
    "gen_ai.tool.message",
    "gen_ai.choice",
]

GENAI_METRIC_TYPES = [
    "gen_ai.client.operation.duration",
    "gen_ai.client.token.usage",
]
