"""Conformance test: OpenInference (Arize) AutoGen instrumentation.

Exercises: agent_run
against a mock OpenAI server, with the OpenInference autogen-agentchat instrumentation.
"""

import asyncio
import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor
    from openinference.instrumentation.config import TraceConfig

    AutogenAgentChatInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


def run_agent():
    """Scenario: basic agent execution via autogen-agentchat."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    print("  [agent_run] basic AutoGen agent execution")

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )

    agent = AssistantAgent(
        name="test_agent",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )

    async def _run():
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken

        response = await agent.on_messages(
            [TextMessage(content="Say hello.", source="user")],
            cancellation_token=CancellationToken(),
        )
        print(f"    -> {str(response.chat_message.content)[:60]}")

    asyncio.run(_run())


def main():
    print("=== OpenInference: AutoGen Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    run_agent()

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
