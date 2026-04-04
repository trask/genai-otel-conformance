"""Conformance test: prototype instrumentation for DSPy.

Exercises: chat via DSPy LM
against a mock OpenAI server, with manual OTel spans.
"""

import json
import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat via DSPy LM with prototype instrumentation."""
    print("  [chat] basic chat via DSPy LM (prototype)")
    import dspy

    request_model = "gpt-4o-mini"
    lm_model = f"openai/{request_model}"
    prompt_text = "Say hello."
    lm = dspy.LM(
        model=lm_model,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        cache=False,
    )
    dspy.configure(lm=lm)

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.response.model", request_model)
        result = lm(prompt_text)
        history_entry = lm.history[-1] if lm.history else None
        if history_entry is not None:
            request_messages = history_entry.get("messages") or []
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps([
                    {
                        "role": message["role"],
                        "parts": [{"type": "text", "content": message["content"]}],
                    }
                    for message in request_messages
                    if isinstance(message.get("content"), str)
                ]),
            )
            response = history_entry.get("response")
            if response is not None:
                if getattr(response, "id", None):
                    span.set_attribute("gen_ai.response.id", response.id)
                finish_reasons = [
                    str(choice.finish_reason).lower()
                    for choice in response.choices
                    if getattr(choice, "finish_reason", None)
                ]
                if finish_reasons:
                    span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
                span.set_attribute(
                    "gen_ai.output.messages",
                    json.dumps([
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": choice.message.content}],
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response.choices
                        if getattr(choice.message, "content", None)
                    ]),
                )
        usage = lm.history[-1].get("usage", {}) if lm.history else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
        if completion_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
        print(f"    -> {str(result)[:60]}")


def run_tool_call():
    """Scenario: tool calling via DSPy ReAct with prototype instrumentation."""
    print("  [chat_tool_call] tool calling via DSPy ReAct (prototype)")
    import dspy

    request_model = "gpt-4o-mini"
    lm_model = f"openai/{request_model}"
    prompt_text = "What's the weather in Seattle?"
    lm = dspy.LM(
        model=lm_model,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        cache=False,
    )
    dspy.configure(lm=lm)
    messages = [{"role": "user", "content": prompt_text}]
    tool_definition = {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "type": "object",
        },
    }
    request_tool = {
        "type": "function",
        "function": tool_definition,
    }

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([request_tool]))
        result = lm(
            messages=messages,
            tools=[request_tool],
        )
        history_entry = lm.history[-1] if lm.history else None
        if history_entry is not None:
            request_messages = history_entry.get("messages") or []
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps([
                    {
                        "role": message["role"],
                        "parts": [{"type": "text", "content": message["content"]}],
                    }
                    for message in request_messages
                    if isinstance(message.get("content"), str)
                ]),
            )
            response = history_entry.get("response")
            if response is not None:
                if getattr(response, "id", None):
                    span.set_attribute("gen_ai.response.id", response.id)
                finish_reasons = [
                    str(choice.finish_reason).lower()
                    for choice in response.choices
                    if getattr(choice, "finish_reason", None)
                ]
                if finish_reasons:
                    span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
                span.set_attribute(
                    "gen_ai.output.messages",
                    json.dumps([
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": choice.message.content}],
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response.choices
                        if getattr(choice.message, "content", None)
                    ]),
                )
        usage = lm.history[-1].get("usage", {}) if lm.history else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
        if completion_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
        print(f"    -> {str(result)[:60]}")


def main():
    print("=== Prototype: DSPy Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    run_chat()
    run_tool_call()

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)
    print("Done.")


if __name__ == "__main__":
    main()
