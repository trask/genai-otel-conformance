"""Conformance test: OTel contrib opentelemetry-instrumentation-openai-agents-v2.

Exercises: agent run with a simple tool
against a mock OpenAI server, with the official OTel OpenAI Agents instrumentation.
"""

import asyncio
import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()


async def run_agent():
    """Run a simple agent with the OpenAI Agents SDK."""
    from agents import Agent, Runner, function_tool
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    import openai

    @function_tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72\u00b0F"

    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    model = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client)

    agent = Agent(
        name="test-agent",
        instructions="You are a helpful assistant.",
        model=model,
        tools=[get_weather],
    )

    print("  [agent_run] agent with tool calling")
    result = await Runner.run(agent, "What's the weather in Seattle?")
    print(f"    -> {str(result.final_output)[:60]}")


def main():
    print("=== OTel Contrib: OpenAI Agents Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    asyncio.run(run_agent())

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
