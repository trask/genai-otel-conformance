"""Shared test infrastructure for CrewAI conformance tests."""

import os

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

tracer = trace.get_tracer("gen_ai.memory.crewai")


def run_crew():
    print("  [crew] basic crew task execution")
    from crewai import Agent, Task, Crew
    from crewai.tools import tool

    os.environ["OPENAI_API_KEY"] = "mock-key"
    os.environ["OPENAI_API_BASE"] = MOCK_BASE_URL

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    researcher = Agent(
        role="Researcher",
        goal="Find information",
        backstory="You are a helpful research assistant.",
        tools=[get_weather],
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description="What's the weather in Seattle?",
        expected_output="The current weather.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    result = crew.kickoff()
    print(f"    -> {str(result)[:60]}")


def run_memory_operations():
    """Scenario: CrewAI memory operations (remember, recall, forget)."""
    print("  [memory] CrewAI memory operations")
    try:
        from crewai.memory import Memory

        os.environ["OPENAI_API_KEY"] = "mock-key"
        os.environ["OPENAI_API_BASE"] = MOCK_BASE_URL

        memory = Memory()

        # remember (update_memory)
        print("    [memory] remember")
        with tracer.start_as_current_span("update_memory", kind=SpanKind.CLIENT) as span:
            span.set_attribute("gen_ai.operation.name", "update_memory")
            span.set_attribute("gen_ai.provider.name", "crewai")
            span.set_attribute("gen_ai.memory.record.content", "The user prefers dark mode.")
            try:
                memory.remember(
                    content="The user prefers dark mode.",
                    scope="user",
                    importance=0.9,
                )
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute("error.type", type(exc).__qualname__)
                print(f"    [memory] remember failed (expected if API not ready): {exc}")

        # recall (search_memory)
        print("    [memory] recall")
        with tracer.start_as_current_span("search_memory", kind=SpanKind.CLIENT) as span:
            span.set_attribute("gen_ai.operation.name", "search_memory")
            span.set_attribute("gen_ai.provider.name", "crewai")
            span.set_attribute("gen_ai.memory.query.text", "user preferences")
            try:
                results = memory.recall(
                    query="user preferences",
                    scope="user",
                )
                if isinstance(results, (list, tuple)):
                    span.set_attribute("gen_ai.memory.search.result.count", len(results))
                print(f"    -> recall results: {results}")
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute("error.type", type(exc).__qualname__)
                print(f"    [memory] recall failed (expected if API not ready): {exc}")

        # forget (delete_memory)
        print("    [memory] forget")
        with tracer.start_as_current_span("delete_memory", kind=SpanKind.CLIENT) as span:
            span.set_attribute("gen_ai.operation.name", "delete_memory")
            span.set_attribute("gen_ai.provider.name", "crewai")
            span.set_attribute("gen_ai.memory.scope", "user")
            try:
                memory.forget(scope="user")
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute("error.type", type(exc).__qualname__)
                print(f"    [memory] forget failed (expected if API not ready): {exc}")

    except ImportError:
        print("    [memory] crewai.memory not available in this version, skipping")
    except Exception as exc:
        print(f"    [memory] unexpected error: {exc}")


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    for scenario in scenarios:
        scenario()

    flush_and_shutdown(tp, lp, mp)
