"""Conformance test: OpenLLMetry (Traceloop) CrewAI instrumentation.

Exercises: agent task execution
against a mock OpenAI server, with the Traceloop CrewAI instrumentation.
"""

from common import run, run_crew


def instrument():
    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
    CrewAIInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OpenLLMetry: CrewAI Conformance Test",
        instrument,
        [run_crew],
    )
