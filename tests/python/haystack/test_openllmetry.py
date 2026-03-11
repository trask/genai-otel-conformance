"""Conformance test: OpenLLMetry (Traceloop) Haystack instrumentation.

Exercises: chat via Haystack pipeline
against a mock OpenAI server, with the Traceloop Haystack instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from opentelemetry.instrumentation.haystack import HaystackInstrumentor
    HaystackInstrumentor().instrument()


def run_chat():
    print("  [chat] basic chat via Haystack pipeline")
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    os.environ["OPENAI_API_KEY"] = "mock-key"

    generator = OpenAIChatGenerator(
        model="gpt-4o-mini",
        api_base_url=MOCK_BASE_URL,
    )

    messages = [ChatMessage.from_user("Say hello.")]
    result = generator.run(messages=messages)
    reply = result["replies"][0]
    print(f"    -> {reply.text[:60]}")


def main():
    print("=== OpenLLMetry: Haystack Conformance Test ===")

    # Pre-load slow haystack modules before connecting OTel to weaver,
    # otherwise weaver's inactivity timeout fires during the long import.
    import haystack  # noqa: F401

    tp, lp, mp = setup_otel()
    instrument()

    run_chat()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
