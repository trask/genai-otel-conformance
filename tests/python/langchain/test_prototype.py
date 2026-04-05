"""Conformance test: prototype instrumentation for LangChain."""

import json
import os

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def _usage_value(usage, key):
    if not usage:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def run_chat_prototype(llm, request_model):
    """Scenario: basic chat completion with prototype instrumentation."""
    print("  [chat] basic chat completion (prototype)")
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        prompt_text = "Say hello."
        resp = llm.invoke(prompt_text)
        meta = getattr(resp, "response_metadata", {})
        if meta.get("model_name"):
            span.set_attribute("gen_ai.response.model", meta["model_name"])
        if getattr(resp, "id", None):
            span.set_attribute("gen_ai.response.id", resp.id)
        if meta.get("finish_reason"):
            span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
        usage = getattr(resp, "usage_metadata", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.input.messages": json.dumps([
                {"role": "user", "parts": [{"type": "text", "content": prompt_text}]}
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": resp.content}],
                    "finish_reason": meta.get("finish_reason"),
                }
            ]),
        }
        if meta.get("model_name"):
            event_attrs["gen_ai.response.model"] = meta["model_name"]
        if getattr(resp, "id", None):
            event_attrs["gen_ai.response.id"] = resp.id
        if meta.get("finish_reason"):
            event_attrs["gen_ai.response.finish_reasons"] = [meta["finish_reason"]]
        if input_tokens is not None:
            event_attrs["gen_ai.usage.input_tokens"] = input_tokens
        if output_tokens is not None:
            event_attrs["gen_ai.usage.output_tokens"] = output_tokens
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {resp.content[:60]}")


def run_chat_streaming_prototype(llm, request_model):
    """Scenario: streaming chat completion with prototype instrumentation."""
    print("  [chat_streaming] streaming chat completion (prototype)")
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        text = ""
        full = None
        for chunk in llm.stream("Tell me a joke."):
            text += chunk.content
            full = chunk if full is None else full + chunk
        if full:
            meta = getattr(full, "response_metadata", {})
            if meta.get("model_name"):
                span.set_attribute("gen_ai.response.model", meta["model_name"])
            if getattr(full, "id", None):
                span.set_attribute("gen_ai.response.id", full.id)
            if meta.get("finish_reason"):
                span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
            usage = getattr(full, "usage_metadata", None)
            input_tokens = _usage_value(usage, "input_tokens")
            output_tokens = _usage_value(usage, "output_tokens")
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        print(f"    -> {text[:60]}")


def run_agent_prototype(llm, request_model):
    """Scenario: agent with tool calling and prototype instrumentation."""
    print("  [agent] agent with tool calling (prototype)")
    from langchain_core.tools import tool

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    tool_definition = {
        "type": "function",
        "name": get_weather.name,
        "description": get_weather.description,
        "parameters": get_weather.args_schema.model_json_schema(),
    }

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([tool_definition]))

        llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")
        resp = llm_with_tools.invoke("What's the weather in Seattle?")
        meta = getattr(resp, "response_metadata", {})
        if meta.get("model_name"):
            span.set_attribute("gen_ai.response.model", meta["model_name"])
        if getattr(resp, "id", None):
            span.set_attribute("gen_ai.response.id", resp.id)
        if meta.get("finish_reason"):
            span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
        usage = getattr(resp, "usage_metadata", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        tool_calls = getattr(resp, "tool_calls", [])
        if tool_calls:
            print(f"    -> tool_call: {tool_calls[0]['name']}")
        else:
            print(f"    -> {str(getattr(resp, 'content', ''))[:60]}")


def run_embeddings_prototype():
    """Scenario: embedding generation with prototype instrumentation."""
    print("  [embeddings] embedding generation (prototype)")
    from langchain_openai import OpenAIEmbeddings

    request_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(
        model=request_model,
        openai_api_base=MOCK_BASE_URL,
        openai_api_key="mock-key",
    )
    with _prototype_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        result = embeddings.embed_query("Hello, world!")
        print(f"    -> embedding dim: {len(result)}")


def main():
    print("=== Prototype: LangChain Conformance Test ===")

    tp, lp, mp = setup_otel()

    from langchain_openai import ChatOpenAI

    request_model = "gpt-4o-mini"
    llm = ChatOpenAI(
        model=request_model,
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )

    run_chat_prototype(llm, request_model)
    run_chat_streaming_prototype(llm, request_model)
    run_agent_prototype(llm, request_model)
    run_embeddings_prototype()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
