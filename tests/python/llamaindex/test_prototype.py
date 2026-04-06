"""Conformance test: prototype instrumentation for LlamaIndex."""

import json
import os

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat_prototype(llm, request_model, request_temperature):
    """Scenario: basic chat completion with prototype instrumentation."""
    from llama_index.core.llms import ChatMessage, MessageRole

    print("  [chat] basic chat completion (prototype)")
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.request.temperature", request_temperature)
        user_content = "Say hello."
        resp = llm.chat([ChatMessage(role=MessageRole.USER, content=user_content)])
        raw = getattr(resp, "raw", None)
        if raw:
            if getattr(raw, "model", None):
                span.set_attribute("gen_ai.response.model", raw.model)
            if getattr(raw, "id", None):
                span.set_attribute("gen_ai.response.id", raw.id)
            if getattr(raw, "choices", None):
                span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in raw.choices])
            if getattr(raw, "usage", None) and raw.usage:
                span.set_attribute("gen_ai.usage.input_tokens", raw.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", raw.usage.completion_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.input.messages": json.dumps([
                {"role": "user", "parts": [{"type": "text", "content": user_content}]}
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": str(resp)}],
                    "finish_reason": raw.choices[0].finish_reason if raw and getattr(raw, "choices", None) else None,
                }
            ]),
        }
        if raw:
            if getattr(raw, "model", None):
                event_attrs["gen_ai.response.model"] = raw.model
            if getattr(raw, "id", None):
                event_attrs["gen_ai.response.id"] = raw.id
            if getattr(raw, "choices", None):
                event_attrs["gen_ai.response.finish_reasons"] = [c.finish_reason for c in raw.choices]
            if getattr(raw, "usage", None) and raw.usage:
                event_attrs["gen_ai.usage.input_tokens"] = raw.usage.prompt_tokens
                event_attrs["gen_ai.usage.output_tokens"] = raw.usage.completion_tokens
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {str(resp)[:60]}")


def run_chat_streaming_prototype(llm, request_model, request_temperature):
    """Scenario: streaming chat completion with prototype instrumentation."""
    from llama_index.core.llms import ChatMessage, MessageRole

    print("  [chat_streaming] streaming chat completion (prototype)")
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.request.temperature", request_temperature)
        text = ""
        stream_resp = llm.stream_chat(
            [ChatMessage(role=MessageRole.USER, content="Tell me a joke.")]
        )
        for token in stream_resp:
            text += token.delta
        raw = getattr(stream_resp, "raw", None)
        if raw:
            if getattr(raw, "model", None):
                span.set_attribute("gen_ai.response.model", raw.model)
            if getattr(raw, "id", None):
                span.set_attribute("gen_ai.response.id", raw.id)
        print(f"    -> {text[:60]}")


def run_agent_prototype(llm, request_model, request_temperature):
    """Scenario: agent with tool calling and prototype instrumentation."""
    print("  [chat_tool_call] agent with tool calling (prototype)")
    from llama_index.core.tools import FunctionTool

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    weather_tool = FunctionTool.from_defaults(fn=get_weather)
    tool_definition = {
        "name": weather_tool.metadata.name,
        "description": weather_tool.metadata.description,
        "fn_schema": weather_tool.metadata.fn_schema.model_json_schema(),
    }
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.request.temperature", request_temperature)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([tool_definition]))

        response = llm.predict_and_call(
            tools=[weather_tool],
            user_msg="What's the weather in Seattle?",
            verbose=False,
        )
        print(f"    -> {str(response)[:60]}")


def run_embeddings_prototype():
    """Scenario: embedding generation with prototype instrumentation."""
    print("  [embeddings] embedding generation (prototype)")
    from llama_index.embeddings.openai import OpenAIEmbedding

    request_model = "text-embedding-3-small"
    embed_model = OpenAIEmbedding(
        model_name=request_model,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )
    with _prototype_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        result = embed_model.get_text_embedding("Hello, world!")
        print(f"    -> embedding dim: {len(result)}")


def main():
    print("=== Prototype: LlamaIndex Conformance Test ===")

    tp, lp, mp = setup_otel()

    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    request_model = "gpt-4o-mini"
    request_temperature = 0.1
    llm = LlamaOpenAI(
        model=request_model,
        temperature=request_temperature,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )

    run_chat_prototype(llm, request_model, request_temperature)
    run_chat_streaming_prototype(llm, request_model, request_temperature)
    run_agent_prototype(llm, request_model, request_temperature)
    run_embeddings_prototype()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
