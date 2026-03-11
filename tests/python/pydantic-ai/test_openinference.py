"""Conformance test: OpenInference (Arize) Pydantic AI instrumentation.

Exercises: chat via Pydantic AI Agent
against a mock OpenAI server, with the OpenInference Pydantic AI instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def add_span_processor(tp):
    """Add the OpenInference pydantic-ai span processor to the tracer provider."""
    from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

    tp.add_span_processor(OpenInferenceSpanProcessor())


def run_chat():
    print("  [chat] basic chat via Pydantic AI Agent")
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url=MOCK_BASE_URL, api_key="mock-key")
    model = OpenAIChatModel("gpt-4o-mini", provider=provider)
    agent = Agent(model, system_prompt="You are a helpful assistant.")

    result = agent.run_sync("Say hello.")
    print(f"    -> {str(result.response)[:60]}")


def main():
    print("=== OpenInference: Pydantic AI Conformance Test ===")

    tp, lp, mp = setup_otel()
    add_span_processor(tp)

    run_chat()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
