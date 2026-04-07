"""Shared test infrastructure for Mem0 conformance tests."""

import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]
_parsed = urlparse(MOCK_BASE_URL)
_SERVER_ADDRESS = _parsed.hostname or "localhost"
_SERVER_PORT = _parsed.port or 80

tracer = trace.get_tracer("gen_ai.memory.mem0")


def _set_common_attrs(span, operation_name):
    """Set attributes common to all Mem0 memory spans."""
    span.set_attribute("gen_ai.operation.name", operation_name)
    span.set_attribute("gen_ai.provider.name", "mem0")
    span.set_attribute("server.address", _SERVER_ADDRESS)
    span.set_attribute("server.port", _SERVER_PORT)


def run_add_memory():
    """Scenario: add a memory record via Mem0 MemoryClient."""
    print("  [memory] add memory")
    from mem0 import MemoryClient

    client = MemoryClient(api_key="mock-key", host=MOCK_BASE_URL)

    messages = [
        {"role": "user", "content": "I'm a vegetarian and allergic to nuts."},
        {"role": "assistant", "content": "Got it! I'll remember your dietary preferences."},
    ]

    with tracer.start_as_current_span("update_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "update_memory")
        span.set_attribute("gen_ai.memory.scope", "user")
        content_text = "; ".join(f"{m['role']}: {m['content']}" for m in messages)
        span.set_attribute("gen_ai.memory.record.content", content_text)
        span.set_attribute("gen_ai.memory.expiration_date", "2027-12-31")
        try:
            result = client.add(messages, user_id="test_user", expiration_date="2027-12-31")
            results = result.get("results", [])
            if results:
                span.set_attribute("gen_ai.memory.record.id", results[0].get("id", ""))
            print(f"    -> add result: {result}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def run_search_memory():
    """Scenario: search memory records via Mem0 MemoryClient."""
    print("  [memory] search memory")
    from mem0 import MemoryClient

    client = MemoryClient(api_key="mock-key", host=MOCK_BASE_URL)

    query = "What are my dietary restrictions?"
    with tracer.start_as_current_span("search_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "search_memory")
        span.set_attribute("gen_ai.memory.scope", "user")
        span.set_attribute("gen_ai.memory.query.text", query)
        threshold = 0.5
        span.set_attribute("gen_ai.memory.search.similarity.threshold", threshold)
        try:
            results = client.search(query, user_id="test_user", threshold=threshold)
            result_items = results.get("results", [])
            span.set_attribute("gen_ai.memory.search.result.count", len(result_items))
            print(f"    -> search results: {len(result_items)} items")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def run_delete_memory():
    """Scenario: delete a memory record via Mem0 MemoryClient."""
    print("  [memory] delete memory")
    from mem0 import MemoryClient

    client = MemoryClient(api_key="mock-key", host=MOCK_BASE_URL)

    # Add a memory first so we have something to delete
    messages = [
        {"role": "user", "content": "My favorite color is blue."},
    ]
    add_result = client.add(messages, user_id="test_user")
    results = add_result.get("results", [])
    if results:
        memory_id = results[0].get("id", "mem_mock_001")
        with tracer.start_as_current_span("delete_memory", kind=SpanKind.CLIENT) as span:
            _set_common_attrs(span, "delete_memory")
            span.set_attribute("gen_ai.memory.scope", "user")
            span.set_attribute("gen_ai.memory.record.id", memory_id)
            try:
                result = client.delete(memory_id)
                print(f"    -> delete result: {result}")
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute("error.type", type(exc).__qualname__)
                raise


def run_delete_all_memories():
    """Scenario: delete all memories for a user via Mem0 MemoryClient."""
    print("  [memory] delete all memories")
    from mem0 import MemoryClient

    client = MemoryClient(api_key="mock-key", host=MOCK_BASE_URL)

    with tracer.start_as_current_span("delete_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "delete_memory")
        span.set_attribute("gen_ai.memory.scope", "user")
        try:
            result = client.delete_all(user_id="test_user")
            print(f"    -> delete all result: {result}")
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
