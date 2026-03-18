"""Shared test infrastructure for Google ADK conformance tests."""

import asyncio
import json
import os
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.sdk.trace import SpanProcessor

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

tracer = trace.get_tracer("gen_ai.memory.google_adk")


class SpanCounter(SpanProcessor):
    """Lightweight span counter for diagnosing whether instrumentation fires."""

    def __init__(self):
        self.count = 0

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self.count += 1

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


def run_agent():
    """Scenario: basic agent execution via Google ADK."""
    # Lazy imports — instrumentation must be active before ADK modules load
    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    print("  [agent_run] basic ADK agent execution")

    # ADK's Gemini creates its own Client internally;
    # it reads GOOGLE_API_KEY from the environment.
    os.environ.setdefault("GOOGLE_API_KEY", "mock-key")

    agent = Agent(
        name="test_agent",
        model=Gemini(model="gemini-2.0-flash", base_url=MOCK_BASE_URL),
        instruction="You are a helpful assistant.",
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="test_app", session_service=session_service)

    event_count = 0

    async def _run():
        nonlocal event_count
        session = await session_service.create_session(
            app_name="test_app", user_id="test_user",
        )
        try:
            async for event in runner.run_async(
                user_id="test_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text="Say hello.")],
                ),
            ):
                event_count += 1
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text:
                        print(f"    -> {text[:60]}")
        except Exception as exc:
            print(f"    [error] agent execution failed: {exc}")

    asyncio.run(_run())
    print(f"    [diagnostic] events received from runner: {event_count}")


# VertexAI RAG can't use the normal HTTP mock server because:
# - rag.upload_file() uses google.auth AuthorizedSession with a hardcoded Google endpoint
# - rag.retrieval_query() uses gRPC, not HTTP
# So we mock the rag module at the Python level instead.


def _mock_rag_module(session_events):
    """Create a mock rag module that simulates VertexAI RAG responses."""
    mock_rag = MagicMock()

    # rag.upload_file — return value is unused by add_session_to_memory
    mock_rag.upload_file.return_value = SimpleNamespace(name="mock-rag-file")

    # rag.retrieval_query — return a response with contexts matching the stored events
    def mock_retrieval_query(text, rag_resources=None, rag_corpora=None,
                             similarity_top_k=None, vector_distance_threshold=None):
        contexts = []
        for display_name, events in session_events.items():
            lines = []
            for event in events:
                if event.content and event.content.parts:
                    text_parts = [p.text for p in event.content.parts if p.text]
                    if text_parts:
                        lines.append(json.dumps({
                            "author": event.author,
                            "timestamp": getattr(event, "timestamp", 0),
                            "text": ".".join(text_parts),
                        }))
            contexts.append(SimpleNamespace(
                source_display_name=display_name,
                text="\n".join(lines),
            ))
        return SimpleNamespace(contexts=SimpleNamespace(contexts=contexts))

    mock_rag.retrieval_query.side_effect = mock_retrieval_query
    return mock_rag


def run_memory_operations():
    """Scenario: Google ADK memory operations using VertexAiRagMemoryService."""
    from google.genai import types
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.adk.memory import InMemoryMemoryService, VertexAiRagMemoryService
    from google.adk.sessions import InMemorySessionService

    print("  [memory] Google ADK memory operations")

    session_service = InMemorySessionService()

    async def _run():
        # Create a session with events to add to memory
        session = await session_service.create_session(
            app_name="test_app", user_id="test_user",
        )
        session.events = [
            Event(
                invocation_id="inv-001",
                author="user",
                content=types.Content(
                    role="user",
                    parts=[types.Part(text="I'm a vegetarian and allergic to nuts.")],
                ),
                actions=EventActions(),
            ),
            Event(
                invocation_id="inv-001",
                author="model",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Got it! I'll remember your dietary preferences.")],
                ),
                actions=EventActions(),
            ),
        ]

        # update_memory using InMemoryMemoryService (no backend needed)
        memory_service = InMemoryMemoryService()
        print("    [memory] add_session_to_memory")
        with tracer.start_as_current_span("update_memory", kind=SpanKind.CLIENT) as span:
            span.set_attribute("gen_ai.operation.name", "update_memory")
            span.set_attribute("gen_ai.provider.name", "google_adk")
            span.set_attribute("gen_ai.memory.record.content",
                               "user: I'm a vegetarian and allergic to nuts.; "
                               "model: Got it! I'll remember your dietary preferences.")
            span.set_attribute("gen_ai.conversation.id", session.id)
            try:
                await memory_service.add_session_to_memory(session)
                print("    -> session added to memory")
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute("error.type", type(exc).__qualname__)
                raise

        # search_memory using VertexAiRagMemoryService (mocked) to demonstrate
        # similarity.threshold capture — only this service exposes vector_distance_threshold
        display_name = f"{session.app_name}.{session.user_id}.{session.id}"
        mock_rag = _mock_rag_module({display_name: session.events})

        fake_vertexai_mod = ModuleType("google.adk.dependencies.vertexai")
        fake_vertexai_mod.rag = mock_rag

        rag_memory_service = VertexAiRagMemoryService(vector_distance_threshold=0.7)

        query = "dietary restrictions"
        print("    [memory] search_memory")
        with patch.dict(sys.modules, {"google.adk.dependencies.vertexai": fake_vertexai_mod}):
            with tracer.start_as_current_span("search_memory", kind=SpanKind.CLIENT) as span:
                span.set_attribute("gen_ai.operation.name", "search_memory")
                span.set_attribute("gen_ai.provider.name", "google_adk")
                span.set_attribute("gen_ai.memory.query.text", query)
                # Threshold from service config — directly observable by native instrumentation
                span.set_attribute("gen_ai.memory.search.similarity.threshold",
                                   rag_memory_service._vertex_rag_store.vector_distance_threshold)
                try:
                    result = await rag_memory_service.search_memory(
                        app_name="test_app", user_id="test_user", query=query,
                    )
                    span.set_attribute("gen_ai.memory.search.result.count", len(result.memories))
                    print(f"    -> search found {len(result.memories)} memories")
                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.set_attribute("error.type", type(exc).__qualname__)
                    raise

    asyncio.run(_run())


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()

    span_counter = SpanCounter()
    tp.add_span_processor(span_counter)

    instrument_fn()

    for scenario in scenarios:
        scenario()

    print(f"\n  [diagnostic] Spans generated: {span_counter.count}")

    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)
