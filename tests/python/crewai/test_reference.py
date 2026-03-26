"""Conformance test: reference instrumentation for CrewAI.

Exercises: agent task execution
against a mock OpenAI server, with manual OTel spans.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_crew():
    """Scenario: basic crew task execution with reference instrumentation."""
    print("  [crew] basic crew task execution (reference)")
    from crewai import Agent, Task, Crew
    from crewai.tools import tool

    request_model = "gpt-4o-mini"
    os.environ["OPENAI_API_KEY"] = "mock-key"
    os.environ["OPENAI_API_BASE"] = MOCK_BASE_URL
    os.environ["OPENAI_MODEL_NAME"] = request_model

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

    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        result = crew.kickoff()
        print(f"    -> {str(result)[:60]}")


def main():
    print("=== Reference: CrewAI Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – reference instrumentation only

    run_crew()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
