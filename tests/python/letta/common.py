"""Shared test infrastructure for Letta conformance tests."""

import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]
_parsed = urlparse(MOCK_BASE_URL)
_SERVER_ADDRESS = _parsed.hostname or "localhost"
_SERVER_PORT = _parsed.port or 80

tracer = trace.get_tracer("gen_ai.memory.letta")


def _attr(obj, name, default=None):
    """Get attribute from either typed SDK object or raw dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_common_attrs(span, operation_name):
    """Set attributes common to all Letta memory spans."""
    span.set_attribute("gen_ai.operation.name", operation_name)
    span.set_attribute("gen_ai.provider.name", "letta")
    span.set_attribute("server.address", _SERVER_ADDRESS)
    span.set_attribute("server.port", _SERVER_PORT)


def _create_agent():
    """Helper: create a Letta agent with memory blocks (setup, not traced)."""
    from letta_client import Letta

    client = Letta(api_key="mock-key", base_url=MOCK_BASE_URL)
    agent_state = client.agents.create(
        memory_blocks=[
            {"label": "human", "value": "The user's name is Alice."},
            {"label": "persona", "value": "I am a helpful assistant named Sam."},
        ],
    )
    return client, agent_state


def run_update_memory():
    """Scenario: update a core memory block via Letta."""
    print("  [memory] update core memory block")

    client, agent_state = _create_agent()
    agent_id = _attr(agent_state, "id")

    # Find the 'human' block
    block_label = "human"
    new_value = "The user's name is Alice. She is a vegetarian and allergic to nuts."

    with tracer.start_as_current_span("update_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "update_memory")
        span.set_attribute("gen_ai.memory.store.id", block_label)
        span.set_attribute("gen_ai.memory.record.content", new_value)
        span.set_attribute("gen_ai.agent.id", agent_id)
        try:
            result = client.agents.blocks.update(
                block_label,
                agent_id=agent_id,
                value=new_value,
            )
            span.set_attribute("gen_ai.memory.record.id", _attr(result, "id", ""))
            print(f"    -> block updated: {_attr(result, 'label')} ({_attr(result, 'id')})")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def run_search_memory():
    """Scenario: search archival memory via Letta."""
    print("  [memory] search archival memory")

    client, agent_state = _create_agent()
    agent_id = _attr(agent_state, "id")

    # First, add a passage to archival memory so there's something to search
    client.agents.passages.create(agent_id, text="Alice is a vegetarian and allergic to nuts.")

    query = "What are the user's dietary restrictions?"
    with tracer.start_as_current_span("search_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "search_memory")
        span.set_attribute("gen_ai.memory.query.text", query)
        span.set_attribute("gen_ai.agent.id", agent_id)
        try:
            results = client.agents.passages.search(agent_id, query=query)
            span.set_attribute("gen_ai.memory.search.result.count", len(results))
            print(f"    -> search results: {len(results)} items")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def run_delete_memory():
    """Scenario: delete an archival memory passage via Letta."""
    print("  [memory] delete archival memory passage")

    client, agent_state = _create_agent()
    agent_id = _attr(agent_state, "id")

    # Add a passage first so we have something to delete
    passage = client.agents.passages.create(
        agent_id, text="My favorite color is blue."
    )
    passage_id = _attr(passage, "id", "")

    with tracer.start_as_current_span("delete_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "delete_memory")
        span.set_attribute("gen_ai.memory.scope", "agent")
        span.set_attribute("gen_ai.memory.record.id", passage_id)
        span.set_attribute("gen_ai.agent.id", agent_id)
        try:
            result = client.agents.passages.delete(passage_id, agent_id=agent_id)
            print(f"    -> delete result: {result}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    for scenario in scenarios:
        scenario()

    flush_and_shutdown(tp, lp, mp)
