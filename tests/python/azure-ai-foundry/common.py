"""Shared test infrastructure for Azure AI Foundry Agent conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def run_invoke_agent(project_client):
    """Scenario: Create and invoke an agent via Azure AI Foundry Agents API."""
    print("  [invoke_agent] Azure AI Foundry Agent")

    # Create agent
    agent = project_client.agents.create_agent(
        model="gpt-4o-mini",
        name="test-agent",
        instructions="You are a helpful assistant.",
    )
    print(f"    agent_id: {agent.id}")

    # Create thread
    thread = project_client.agents.create_thread()
    print(f"    thread_id: {thread.id}")

    # Send message
    project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content="Say hello.",
    )

    # Create and process run (blocks until complete)
    run = project_client.agents.create_and_process_run(
        thread_id=thread.id,
        agent_id=agent.id,
    )
    print(f"    run status: {run.status}")

    # Get response messages
    messages = project_client.agents.list_messages(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            for content in msg.content:
                if hasattr(content, "text"):
                    print(f"    -> {content.text.value[:60]}")
                    break

    # Clean up
    project_client.agents.delete_agent(agent.id)


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()

    # Enable Azure SDK tracing via environment variable
    os.environ.setdefault("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")

    # Explicitly set the azure-core tracing implementation to OpenTelemetry
    from azure.core.settings import settings as azure_settings
    azure_settings.tracing_implementation = "opentelemetry"

    instrument_fn()

    from azure.ai.projects import AIProjectClient
    from azure.core.credentials import AzureKeyCredential

    project_client = AIProjectClient(
        endpoint=MOCK_BASE_URL,
        credential=AzureKeyCredential("mock-key"),
    )

    for scenario in scenarios:
        scenario(project_client)

    flush_and_shutdown(tp, lp, mp)
