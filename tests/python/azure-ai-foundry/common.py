"""Shared test infrastructure for Azure AI Foundry Agent conformance tests."""

import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]
_parsed = urlparse(MOCK_BASE_URL)
_SERVER_ADDRESS = _parsed.hostname or "localhost"
_SERVER_PORT = _parsed.port or 443

tracer = trace.get_tracer("gen_ai.client.azure_ai_foundry")

AGENT_MODEL = "gpt-4o-mini"
AGENT_NAME = "conformance-test-agent"


class MockCredential:
    """Dummy TokenCredential for testing against the mock server."""

    def get_token(self, *scopes, **kwargs):
        from azure.core.credentials import AccessToken

        return AccessToken("mock-token", 9999999999)


def create_client():
    """Create an Azure AI Projects client pointing at the mock server."""
    from azure.ai.projects import AIProjectClient
    from azure.core.pipeline.policies import SansIOHTTPPolicy

    return AIProjectClient(
        endpoint=MOCK_BASE_URL,
        credential=MockCredential(),
        authentication_policy=SansIOHTTPPolicy(),
    )


def run_invoke_agent(client):
    """Exercise Azure AI Foundry Agents API with manual OTel spans.

    Creates a CLIENT span with gen_ai invoke_agent attributes to demonstrate
    what an instrumentation library should capture for the Azure AI Agents API
    (create agent, create_thread_and_process_run, get result).
    """
    print("  [invoke_agent] Azure AI Foundry Agents: create + run")

    # Create agent
    agent = client.agents.create_agent(
        model=AGENT_MODEL,
        name=AGENT_NAME,
        instructions="You are a helpful assistant.",
    )

    # Create thread, add message, and run — all in one call, wrapped in manual span
    from azure.ai.agents.models import AgentThreadCreationOptions, ThreadMessageOptions

    with tracer.start_as_current_span("invoke_agent", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.provider.name", "azure.ai.openai")
        span.set_attribute("gen_ai.agent.id", agent.id)
        span.set_attribute("gen_ai.agent.name", agent.name or "")
        span.set_attribute("gen_ai.request.model", AGENT_MODEL)
        span.set_attribute("server.address", _SERVER_ADDRESS)
        span.set_attribute("server.port", _SERVER_PORT)
        try:
            run = client.agents.create_thread_and_run(
                agent_id=agent.id,
                thread=AgentThreadCreationOptions(
                    messages=[
                        ThreadMessageOptions(
                            role="user",
                            content="Hello, agent!",
                        ),
                    ],
                ),
            )

            if run.usage:
                span.set_attribute("gen_ai.usage.input_tokens", run.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", run.usage.completion_tokens)

            print(f"    -> run status: {run.status}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # Clean up
    client.agents.delete_agent(agent.id)


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
