"""Shared test infrastructure for OpenAI Assistants conformance tests.

NOTE: The OpenAI Assistants API is deprecated and will shut down on
August 26, 2026. It is being replaced by the Responses API with
Conversations. These tests exercise the legacy Assistants API to verify
instrumentation coverage for existing deployments.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def run_invoke_agent(client):
    """Scenario: Create and invoke an assistant (remote agent invocation)."""
    print("  [invoke_agent] OpenAI Assistants API")

    # Create assistant
    assistant = client.beta.assistants.create(
        model="gpt-4o-mini",
        name="test-assistant",
        instructions="You are a helpful assistant.",
    )
    print(f"    assistant_id: {assistant.id}")

    # Create thread
    thread = client.beta.threads.create()
    print(f"    thread_id: {thread.id}")

    # Send message
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Say hello.",
    )

    # Create and poll run (blocks until complete)
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    print(f"    run status: {run.status}")

    # Get response messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            for content in msg.content:
                if content.type == "text":
                    print(f"    -> {content.text.value[:60]}")
                    break

    # Clean up
    client.beta.assistants.delete(assistant.id)


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    import openai
    client = openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
