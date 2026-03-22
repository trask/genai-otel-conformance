"""Shared test infrastructure for AWS Bedrock Agent conformance tests."""

import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]
_parsed = urlparse(MOCK_BASE_URL)
_SERVER_ADDRESS = _parsed.hostname or "localhost"
_SERVER_PORT = _parsed.port or 443

tracer = trace.get_tracer("gen_ai.client.aws_bedrock")

AGENT_ID = "MOCK_AGENT_ID"
AGENT_ALIAS_ID = "MOCK_ALIAS_ID"
SESSION_ID = "mock-session-001"
AGENT_NAME = "conformance-test-agent"


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
    """Exercise Bedrock Agent Runtime InvokeAgent with manual OTel spans.

    Creates a CLIENT span with gen_ai invoke_agent attributes to demonstrate
    what an instrumentation library should capture.
    """
    print("  [invoke_agent] Bedrock Agent Runtime InvokeAgent")
    with tracer.start_as_current_span("invoke_agent", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")
        span.set_attribute("gen_ai.agent.id", AGENT_ID)
        span.set_attribute("gen_ai.agent.name", AGENT_NAME)
        span.set_attribute("server.address", _SERVER_ADDRESS)
        span.set_attribute("server.port", _SERVER_PORT)
        try:
            response = client.invoke_agent(
                agentId=AGENT_ID,
                agentAliasId=AGENT_ALIAS_ID,
                sessionId=SESSION_ID,
                inputText="Hello, agent!",
            )
            # Consume the event stream
            text = ""
            for event in response["completion"]:
                if "chunk" in event:
                    text += event["chunk"].get("bytes", b"").decode("utf-8")
            print(f"    -> {text[:60]}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_bedrock_agent_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
