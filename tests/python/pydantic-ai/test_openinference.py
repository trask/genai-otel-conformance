"""Conformance test: OpenInference (Arize) Pydantic AI instrumentation.

Exercises: chat via Pydantic AI Agent
against a mock OpenAI server, with the OpenInference Pydantic AI instrumentation.

Pydantic AI emits spans via logfire. We configure logfire with send_to_logfire=False
and add both an OTLP exporter (for Weaver) and the OpenInference span processor
(for OI-specific attributes).
"""

import os

import logfire
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"
OTLP_ENDPOINT = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]


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


def run_chat_tool_call():
    print("  [chat_tool_call] tool calling via Pydantic AI Agent")
    from pydantic_ai import Agent, Tool
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url=MOCK_BASE_URL, api_key="mock-key")
    model = OpenAIChatModel("gpt-4o-mini", provider=provider)

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72\u00b0F"

    agent = Agent(
        model,
        system_prompt="You are a helpful assistant.",
        tools=[Tool(get_weather)],
    )
    result = agent.run_sync("What's the weather in Seattle?")
    print(f"    -> {str(result.response)[:60]}")


def main():
    print("=== OpenInference: Pydantic AI Conformance Test ===")

    otlp_processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
    )
    oi_processor = OpenInferenceSpanProcessor()

    logfire.configure(
        send_to_logfire=False,
        additional_span_processors=[otlp_processor, oi_processor],
    )

    from pydantic_ai import Agent
    Agent.instrument_all()

    run_chat()
    run_chat_tool_call()

    print("Flushing telemetry...")
    otlp_processor.force_flush()
    otlp_processor.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
