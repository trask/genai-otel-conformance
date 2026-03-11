"""Shared test infrastructure for Google ADK conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def run_agent():
    """Scenario: basic agent execution via Google ADK."""
    from google import genai
    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    print("  [agent_run] basic ADK agent execution")

    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )

    agent = Agent(
        name="test_agent",
        model="gemini-2.0-flash",
        instruction="You are a helpful assistant.",
        client=client,
    )

    session_service = InMemorySessionService()
    session = session_service.create_session(app_name="test_app", user_id="test_user")

    runner = Runner(agent=agent, app_name="test_app", session_service=session_service)

    import asyncio

    async def _run():
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Say hello.")],
            ),
        ):
            if event.content and event.content.parts:
                text = event.content.parts[0].text
                if text:
                    print(f"    -> {text[:60]}")
                    return

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
