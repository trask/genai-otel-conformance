"""Conformance test: prototype instrumentation for LiteLLM.

Exercises: chat, chat_streaming, chat_tool_call, embeddings
against a mock OpenAI server, with manual OTel spans.
"""

import json
import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat completion with prototype instrumentation."""
    import litellm
    print("  [chat] basic chat completion via LiteLLM (prototype)")
    request_model = "gpt-4o-mini"
    litellm_model = f"openai/{request_model}"
    prompt_text = "Say hello."
    request_messages = [{"role": "user", "content": prompt_text}]
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = litellm.completion(
            model=litellm_model,
            messages=request_messages,
            api_base=MOCK_BASE_URL,
            api_key="mock-key",
        )
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps([
                {
                    "role": message["role"],
                    "parts": [{"type": "text", "content": message["content"]}],
                }
                for message in request_messages
            ]),
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons",
                           [c.finish_reason for c in resp.choices])
        span.set_attribute(
            "gen_ai.output.messages",
            json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": choice.message.content}],
                    "finish_reason": choice.finish_reason,
                }
                for choice in resp.choices
                if choice.message.content
            ]),
        )
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.response.id": resp.id,
            "gen_ai.response.model": resp.model,
            "gen_ai.response.finish_reasons": [c.finish_reason for c in resp.choices],
            "gen_ai.input.messages": json.dumps([
                {"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]}
                for m in request_messages
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": c.message.content}],
                    "finish_reason": c.finish_reason,
                }
                for c in resp.choices if c.message.content
            ]),
        }
        if resp.usage:
            event_attrs["gen_ai.usage.input_tokens"] = resp.usage.prompt_tokens
            event_attrs["gen_ai.usage.output_tokens"] = resp.usage.completion_tokens
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming():
    """Scenario: streaming chat completion with prototype instrumentation."""
    import litellm
    print("  [chat_streaming] streaming chat via LiteLLM (prototype)")
    request_model = "gpt-4o-mini"
    litellm_model = f"openai/{request_model}"
    prompt_text = "Tell me a joke."
    request_messages = [{"role": "user", "content": prompt_text}]
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = litellm.completion(
            model=litellm_model,
            messages=request_messages,
            api_base=MOCK_BASE_URL,
            api_key="mock-key",
            stream=True,
        )
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps([
                {
                    "role": message["role"],
                    "parts": [{"type": "text", "content": message["content"]}],
                }
                for message in request_messages
            ]),
        )
        text = ""
        finish_reason = None
        for chunk in resp:
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
            if chunk.choices[0].finish_reason is not None:
                finish_reason = chunk.choices[0].finish_reason
        span.set_attribute(
            "gen_ai.output.messages",
            json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": text}],
                    **({"finish_reason": finish_reason} if finish_reason is not None else {}),
                }
            ]),
        )
        print(f"    -> {text[:60]}")


def run_chat_tool_call():
    """Scenario: chat with tool calling with prototype instrumentation."""
    import litellm

    print("  [chat_tool_call] chat with tool calling via LiteLLM (prototype)")
    request_model = "gpt-4o-mini"
    litellm_model = f"openai/{request_model}"
    request_messages = [{"role": "user", "content": "What's the weather in Seattle?"}]
    request_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([{
            "type": request_tool["type"],
            "name": request_tool["function"]["name"],
            "description": request_tool["function"]["description"],
            "parameters": request_tool["function"]["parameters"],
        }]))
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps([
                {
                    "role": message["role"],
                    "parts": [{"type": "text", "content": message["content"]}],
                }
                for message in request_messages
            ]),
        )
        resp = litellm.completion(
            model=litellm_model,
            messages=request_messages,
            tools=[request_tool],
            api_base=MOCK_BASE_URL,
            api_key="mock-key",
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in resp.choices])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
        tool_calls = getattr(resp.choices[0].message, "tool_calls", None)
        if tool_calls:
            print(f"    -> tool_call: {tool_calls[0].function.name}")
        else:
            print(f"    -> {resp.choices[0].message.content[:60]}")


def run_embeddings():
    """Scenario: embedding generation with prototype instrumentation."""
    import litellm
    print("  [embeddings] embedding generation via LiteLLM (prototype)")
    request_model = "text-embedding-3-small"
    litellm_model = f"openai/{request_model}"
    with _prototype_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = litellm.embedding(
            model=litellm_model,
            input=["Hello, world!"],
            api_base=MOCK_BASE_URL,
            api_key="mock-key",
        )
        if resp.model:
            span.set_attribute("gen_ai.response.model", resp.model)
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
        print(f"    -> embedding dim: {len(resp.data[0]['embedding'])}")


def main():
    print("=== Prototype: LiteLLM Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    run_chat()
    run_chat_streaming()
    run_chat_tool_call()
    run_embeddings()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
