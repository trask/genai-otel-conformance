"""Shared test infrastructure for Azure AI Foundry conformance tests."""

import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]
_parsed = urlparse(MOCK_BASE_URL)
_SERVER_ADDRESS = _parsed.hostname or "localhost"
_SERVER_PORT = _parsed.port or 443

tracer = trace.get_tracer("gen_ai.memory.azure_ai_foundry")


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


def run_memory_operations(client):
    """Exercise Azure AI Foundry Memory operations for conformance testing.

    Spans are created manually to demonstrate what an instrumentation library
    should capture.  The real SDK calls go to the mock server.

    Operations exercised:
      1. create_memory_store  (POST /memory_stores)
      2. update_memory        (POST /memory_stores/{name}:update_memories)
      3. search_memory        (POST /memory_stores/{name}:search_memories)
      4. delete_memory         (POST /memory_stores/{name}:delete_scope)
      5. delete_memory_store   (DELETE /memory_stores/{name})
    """
    store_name = "conformance-test-store"

    def _set_common_attrs(span, operation_name):
        span.set_attribute("gen_ai.operation.name", operation_name)
        span.set_attribute("gen_ai.provider.name", "az.ai.foundry")
        span.set_attribute("server.address", _SERVER_ADDRESS)
        span.set_attribute("server.port", _SERVER_PORT)

    # 1. Create memory store
    print("  [create_memory_store] Azure AI Foundry create memory store")
    with tracer.start_as_current_span("create_memory_store", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "create_memory_store")
        span.set_attribute("gen_ai.memory.store.name", store_name)
        span.set_attribute("gen_ai.memory.scope", "conformance-test")
        try:
            from azure.ai.projects.models import MemoryStoreDefaultDefinition

            result = client.beta.memory_stores.create(
                name=store_name,
                definition=MemoryStoreDefaultDefinition(),
            )
            store_id = result.get("id", store_name)
            span.set_attribute("gen_ai.memory.store.id", store_id)
            print(f"    -> created store: {result['name']} ({store_id})")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 2. Update memory (add conversation to memory)
    print("  [update_memory] Azure AI Foundry update memories")
    with tracer.start_as_current_span("update_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "update_memory")
        span.set_attribute("gen_ai.memory.store.name", store_name)
        span.set_attribute("gen_ai.memory.record.content",
                           "user: I'm a vegetarian and allergic to nuts.")
        try:
            poller = client.beta.memory_stores.begin_update_memories(
                name=store_name,
                scope="test-user-001",
                items=[
                    {"role": "user", "type": "message",
                     "content": "I'm a vegetarian and allergic to nuts."},
                    {"role": "assistant", "type": "message",
                     "content": "Got it! I'll remember your dietary preferences."},
                ],
                update_delay=0,
            )
            update_result = poller.result()
            operations = update_result.get("memory_operations", [])
            if operations:
                first_op = operations[0]
                memory_item = first_op.get("memory_item", {})
                span.set_attribute("gen_ai.memory.record.id",
                                   memory_item.get("memory_id", ""))
            print(f"    -> update completed, {len(operations)} operations")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 3. Search memory
    query = "What are the user's dietary restrictions?"
    print("  [search_memory] Azure AI Foundry search memories")
    with tracer.start_as_current_span("search_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "search_memory")
        span.set_attribute("gen_ai.memory.store.name", store_name)
        span.set_attribute("gen_ai.memory.query.text", query)
        try:
            search_result = client.beta.memory_stores.search_memories(
                name=store_name,
                scope="test-user-001",
                items=query,
            )
            memories = search_result.get("memories", [])
            span.set_attribute("gen_ai.memory.search.result.count", len(memories))
            print(f"    -> found {len(memories)} memories")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 4. Delete scope (delete_memory)
    print("  [delete_memory] Azure AI Foundry delete scope")
    with tracer.start_as_current_span("delete_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "delete_memory")
        span.set_attribute("gen_ai.memory.store.name", store_name)
        span.set_attribute("gen_ai.memory.scope", "test-user-001")
        try:
            delete_scope_result = client.beta.memory_stores.delete_scope(
                name=store_name,
                scope="test-user-001",
            )
            print(f"    -> scope deleted: {delete_scope_result.get('deleted', False)}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 5. Delete memory store
    print("  [delete_memory_store] Azure AI Foundry delete memory store")
    with tracer.start_as_current_span("delete_memory_store", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "delete_memory_store")
        span.set_attribute("gen_ai.memory.store.name", store_name)
        try:
            delete_result = client.beta.memory_stores.delete(name=store_name)
            print(f"    -> store deleted: {delete_result.get('deleted', False)}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
