"""Conformance test: OpenInference openinference-instrumentation-openai-agents.

Exercises: agent run with a simple tool
against a mock OpenAI server, with the OpenInference OpenAI Agents instrumentation.
"""

import asyncio
import os

from opentelemetry import trace

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
    OpenAIAgentsInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


async def run_agent():
    """Run a simple agent with the OpenAI Agents SDK."""
    from agents import Agent, Runner
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    import openai

    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    model = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client)

    agent = Agent(
        name="test-agent",
        instructions="You are a helpful assistant.",
        model=model,
    )

    print("  [agent_run] basic agent execution")
    result = await Runner.run(agent, "Say hello.")
    print(f"    -> {str(result.final_output)[:60]}")


def main():
    print("=== OpenInference: OpenAI Agents Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    asyncio.run(run_agent())

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
