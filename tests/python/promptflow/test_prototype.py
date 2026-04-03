"""Conformance test: prototype instrumentation for Promptflow.

Exercises: chat via OpenAI client
against a mock OpenAI server, with manual OTel spans (no @trace decorator).
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat completion with prototype instrumentation."""
    print("  [chat] basic chat completion (prototype)")
    import openai

    client = openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons",
                           [c.finish_reason for c in resp.choices])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
        print(f"    -> {resp.choices[0].message.content[:60]}")


def main():
    print("=== Prototype: Promptflow Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – no @trace decorator, no start_trace()

    run_chat()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
