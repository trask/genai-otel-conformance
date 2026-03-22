"""Shared test infrastructure for AWS Bedrock Agent conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def create_bedrock_agent_client():
    """Create a boto3 Bedrock Agent Runtime client pointing at the mock server."""
    import boto3

    client = boto3.client(
        "bedrock-agent-runtime",
        endpoint_url=MOCK_BASE_URL,
        region_name="us-east-1",
        aws_access_key_id="mock",
        aws_secret_access_key="mock",
    )
    return client


def run_invoke_agent(client):
    """Scenario: Bedrock Agent Runtime InvokeAgent API."""
    print("  [invoke_agent] Bedrock Agent Runtime InvokeAgent")
    response = client.invoke_agent(
        agentId="MOCK_AGENT_ID",
        agentAliasId="MOCK_ALIAS_ID",
        sessionId="mock-session-001",
        inputText="Say hello.",
    )

    # Read the event stream
    completion_text = ""
    for event in response["completion"]:
        if "chunk" in event:
            completion_text += event["chunk"]["bytes"].decode("utf-8")
    print(f"    -> {completion_text[:60]}")


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_bedrock_agent_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
