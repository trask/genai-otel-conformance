"""Conformance test: prototype instrumentation for Haystack.

Exercises: chat via Haystack pipeline
against a mock OpenAI server, with manual OTel spans.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat via Haystack pipeline with prototype instrumentation."""
    print("  [chat] basic chat via Haystack pipeline (prototype)")
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    os.environ["OPENAI_API_KEY"] = "mock-key"
    request_model = "gpt-4o-mini"

    generator = OpenAIChatGenerator(
        model=request_model,
        api_base_url=MOCK_BASE_URL,
    )

    messages = [ChatMessage.from_user("Say hello.")]

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        result = generator.run(messages=messages)
        reply = result["replies"][0]
        if hasattr(reply, "meta") and reply.meta:
            meta = reply.meta
            if "model" in meta:
                span.set_attribute("gen_ai.response.model", meta["model"])
            if "finish_reason" in meta:
                span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
            if "usage" in meta and meta["usage"]:
                usage = meta["usage"]
                if "prompt_tokens" in usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
                if "completion_tokens" in usage:
                    span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])
        print(f"    -> {reply.text[:60]}")


def main():
    print("=== Prototype: Haystack Conformance Test ===")

    # Pre-load slow haystack modules before connecting OTel to weaver,
    # otherwise weaver's inactivity timeout fires during the long import.
    import haystack  # noqa: F401

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    run_chat()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
