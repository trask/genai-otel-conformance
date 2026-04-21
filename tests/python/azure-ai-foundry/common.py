"""Shared test infrastructure for Azure AI Foundry Agent conformance tests."""

import json
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
    what an instrumentation library should capture for the Azure AI Foundry v2
    agent flow (create agent version, invoke through Responses API, get
    result).
    """
    print("  [invoke_agent] Azure AI Foundry Agents: create + run")

    from azure.ai.projects.models import PromptAgentDefinition

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Create agent version using the v2 AIProjectClient surface.
    agent = client.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model=AGENT_MODEL,
            instructions="You are a helpful assistant.",
            tools=tool_defs,
        ),
    )

    # Invoke the agent through the Responses API, wrapped in a manual span.
    openai_client = client.get_openai_client()

    with tracer.start_as_current_span("invoke_agent", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.provider.name", "azure.ai.openai")
        span.set_attribute("gen_ai.agent.id", agent.id)
        span.set_attribute("gen_ai.agent.name", agent.name or "")
        span.set_attribute("gen_ai.request.model", AGENT_MODEL)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([
            {
                "type": t["type"],
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in tool_defs
        ]))
        span.set_attribute("server.address", _SERVER_ADDRESS)
        span.set_attribute("server.port", _SERVER_PORT)
        try:
            response = openai_client.responses.create(
                input="Hello, agent!",
                extra_body={
                    "agent_reference": {
                        "name": agent.name,
                        "type": "agent_reference",
                    }
                },
            )

            if response.usage:
                span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)

            print(f"    -> response id: {response.id}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise
        finally:
            openai_client.close()

    # Clean up
    client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
