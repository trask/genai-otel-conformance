"""Shared test infrastructure for AutoGen conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def run_agent():
    """Scenario: basic agent execution via AutoGen."""
    import asyncio
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


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    for scenario in scenarios:
        scenario()

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)
