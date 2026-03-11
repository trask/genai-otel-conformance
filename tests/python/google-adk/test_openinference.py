"""Conformance test: OpenInference (Arize) Google ADK instrumentation.

Exercises: agent_run
against a mock Google GenAI server, with the OpenInference Google ADK instrumentation.
"""

from opentelemetry import trace

from common import run, run_agent


def instrument():
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    GoogleADKInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run(
        "OpenInference: Google ADK Conformance Test",
        instrument,
        [run_agent],
    )
