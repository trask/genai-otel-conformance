"""Conformance test: prototype instrumentation for Groq."""

import json
import os

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat_prototype(client):
    """Scenario: basic chat completion with prototype instrumentation."""
    print("  [chat] basic chat completion (prototype)")
    request_model = "llama-3.1-8b-instant"
    with _prototype_tracer.start_as_current_span("chat llama-3.1-8b-instant") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "groq")
        span.set_attribute("gen_ai.request.model", request_model)
        messages = [{"role": "user", "content": "Say hello."}]
        resp = client.chat.completions.create(
            model=request_model,
            messages=messages,
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in resp.choices])
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
                for m in messages
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": c.message.role,
                    "parts": [{"type": "text", "content": c.message.content}],
                    "finish_reason": c.finish_reason,
                }
                for c in resp.choices
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


def run_chat_streaming_prototype(client):
    """Scenario: streaming chat completion with prototype instrumentation."""
    print("  [chat_streaming] streaming chat completion (prototype)")
    request_model = "llama-3.1-8b-instant"
    with _prototype_tracer.start_as_current_span("chat llama-3.1-8b-instant") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "groq")
        span.set_attribute("gen_ai.request.model", request_model)
        stream = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            stream=True,
        )
        text = ""
        model = None
        response_id = None
        finish_reasons = []
        for chunk in stream:
            model = model or getattr(chunk, "model", None)
            response_id = response_id or getattr(chunk, "id", None)
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reasons.append(chunk.choices[0].finish_reason)
        if model:
            span.set_attribute("gen_ai.response.model", model)
        if response_id:
            span.set_attribute("gen_ai.response.id", response_id)
        if finish_reasons:
            span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        print(f"    -> {text[:60]}")


def main():
    print("=== Prototype: Groq Conformance Test ===")

    tp, lp, mp = setup_otel()

    import groq

    client = groq.Groq(base_url=MOCK_BASE_URL, api_key="mock-key")

    run_chat_prototype(client)
    run_chat_streaming_prototype(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
