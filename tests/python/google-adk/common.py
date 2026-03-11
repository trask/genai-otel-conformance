"""Shared test infrastructure for Google ADK conformance tests."""

import asyncio
import os
import time

from opentelemetry.sdk.trace import SpanProcessor

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


class SpanCounter(SpanProcessor):
    """Lightweight span counter for diagnosing whether instrumentation fires."""

    def __init__(self):
        self.count = 0

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self.count += 1

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


def run_agent():
    """Scenario: basic agent execution via Google ADK."""
    # Lazy imports — instrumentation must be active before ADK modules load
    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    print("  [agent_run] basic ADK agent execution")

    # ADK's Gemini creates its own Client internally;
    # it reads GOOGLE_API_KEY from the environment.
    os.environ.setdefault("GOOGLE_API_KEY", "mock-key")

    agent = Agent(
        name="test_agent",
        model=Gemini(model="gemini-2.0-flash", base_url=MOCK_BASE_URL),
        instruction="You are a helpful assistant.",
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="test_app", session_service=session_service)

    event_count = 0

    async def _run():
        nonlocal event_count
        session = await session_service.create_session(
            app_name="test_app", user_id="test_user",
        )
        try:
            async for event in runner.run_async(
                user_id="test_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text="Say hello.")],
                ),
            ):
                event_count += 1
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text:
                        print(f"    -> {text[:60]}")
        except Exception as exc:
            print(f"    [error] agent execution failed: {exc}")

    asyncio.run(_run())
    print(f"    [diagnostic] events received from runner: {event_count}")


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()

    span_counter = SpanCounter()
    tp.add_span_processor(span_counter)

    instrument_fn()

    for scenario in scenarios:
        scenario()

    print(f"\n  [diagnostic] Spans generated: {span_counter.count}")

    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)
