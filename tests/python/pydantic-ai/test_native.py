"""Conformance test: Pydantic AI native OTel instrumentation.

Exercises: chat via Pydantic AI Agent
against a mock OpenAI server, with Pydantic AI's built-in OTel tracing
(logfire integration emits standard OTel spans).
"""

import os

import logfire
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

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
    print("=== Native: Pydantic AI Conformance Test ===")

    # Let logfire configure the TracerProvider; add our OTLP exporter as
    # an additional span processor so spans reach weaver.
    otlp_processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
    )
    logfire.configure(
        send_to_logfire=False,
        additional_span_processors=[otlp_processor],
    )

    # Enable OTel instrumentation for all pydantic-ai agents
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
