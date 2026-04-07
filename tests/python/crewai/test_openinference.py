"""Conformance test: OpenInference (Arize) CrewAI instrumentation.

Exercises: agent task execution
against a mock OpenAI server, with the OpenInference CrewAI instrumentation.
"""

from opentelemetry import trace

from common import run, run_crew


def instrument():
    from openinference.instrumentation.crewai import CrewAIInstrumentor
    CrewAIInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run(
        "OpenInference: CrewAI Conformance Test",
        instrument,
        [run_crew],
    )
