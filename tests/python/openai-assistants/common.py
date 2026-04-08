"""Shared test infrastructure for OpenAI Assistants conformance tests."""

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

tracer = trace.get_tracer("gen_ai.client.openai")


def create_client():
    """Create an OpenAI client pointing at the mock server."""
    from openai import OpenAI

    return OpenAI(
        api_key="mock-key",
        base_url=f"{MOCK_BASE_URL}/v1",
    )


def run_invoke_agent(client):
    """Exercise OpenAI Assistants API with manual OTel spans.

    Creates a CLIENT span with gen_ai invoke_agent attributes to demonstrate
    what an instrumentation library should capture for the Assistants API
    (create assistant, create thread, add message, create run, poll, get messages).
    """
    print("  [invoke_agent] OpenAI Assistants: create + run")

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

    # Create assistant
    assistant = client.beta.assistants.create(
        model="gpt-4o-mini",
        name="conformance-test-assistant",
        instructions="You are a helpful assistant.",
        tools=tool_defs,
    )

    # Create thread
    thread = client.beta.threads.create()

    # Add message
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Hello, assistant!",
    )

    # Create run and wrap in manual invoke_agent span
    with tracer.start_as_current_span("invoke_agent", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.agent.id", assistant.id)
        span.set_attribute("gen_ai.agent.name", assistant.name or "")
        span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
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
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )

            # Poll for completion (mock returns completed immediately)
            if run.status != "completed":
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )

            if run.usage:
                span.set_attribute("gen_ai.usage.input_tokens", run.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", run.usage.completion_tokens)

            # Get messages
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            assistant_messages = [m for m in messages.data if m.role == "assistant"]
            if assistant_messages:
                text = assistant_messages[0].content[0].text.value
                print(f"    -> {text[:60]}")
            else:
                print("    -> (no assistant response)")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # Clean up
    client.beta.assistants.delete(assistant.id)


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
