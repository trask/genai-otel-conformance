"""Conformance test: reference instrumentation for LiteLLM.

Exercises: chat, chat_streaming, embeddings
against a mock OpenAI server, with manual OTel spans.
"""

import json
import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_chat():
    """Scenario: basic chat completion with reference instrumentation."""
    import litellm
    print("  [chat] basic chat completion via LiteLLM (reference)")
    request_model = "gpt-4o-mini"
    litellm_model = f"openai/{request_model}"
    prompt_text = "Say hello."
    request_messages = [{"role": "user", "content": prompt_text}]
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
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
        print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming():
    """Scenario: streaming chat completion with reference instrumentation."""
    import litellm
    print("  [chat_streaming] streaming chat via LiteLLM (reference)")
    request_model = "gpt-4o-mini"
    litellm_model = f"openai/{request_model}"
    prompt_text = "Tell me a joke."
    request_messages = [{"role": "user", "content": prompt_text}]
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
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


def run_embeddings():
    """Scenario: embedding generation with reference instrumentation."""
    import litellm
    print("  [embeddings] embedding generation via LiteLLM (reference)")
    request_model = "text-embedding-3-small"
    litellm_model = f"openai/{request_model}"
    with _reference_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
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
    print("=== Reference: LiteLLM Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – reference instrumentation only

    run_chat()
    run_chat_streaming()
    run_embeddings()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
