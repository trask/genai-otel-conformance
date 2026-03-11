"""Conformance test: OpenLLMetry (Traceloop) CrewAI instrumentation.

Exercises: agent task execution
against a mock OpenAI server, with the Traceloop CrewAI instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    CrewAIInstrumentor().instrument()


def run_crew():
    print("  [crew] basic crew task execution")
    from crewai import Agent, Task, Crew

    os.environ["OPENAI_API_KEY"] = "mock-key"
    os.environ["OPENAI_API_BASE"] = MOCK_BASE_URL

    researcher = Agent(
        role="Researcher",
        goal="Find information",
        backstory="You are a helpful research assistant.",
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description="Say hello in a creative way.",
        expected_output="A creative greeting.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=False)
    result = crew.kickoff()
    print(f"    -> {str(result)[:60]}")


def main():
    print("=== OpenLLMetry: CrewAI Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    run_crew()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
