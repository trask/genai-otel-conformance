"""Conformance test: Google ADK native OTel instrumentation.

Google ADK automatically emits OTel spans when a global tracer provider is set.
Exercises: agent_run.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

from google import genai
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def run_agent():
    """Scenario: basic agent execution via Google ADK with native tracing."""
    import asyncio

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


def main():
    print("=== Native: Google ADK Conformance Test ===")

    tp, lp, mp = setup_otel()

    # Google ADK emits OTel spans automatically when a global tracer provider is set
    from opentelemetry import trace as trace_api
    trace_api.set_tracer_provider(tp)

    run_agent()

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)
    print("Done.")


if __name__ == "__main__":
    main()
