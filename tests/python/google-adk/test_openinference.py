"""Conformance test: OpenInference (Arize) Google ADK instrumentation.

Exercises: agent_run
against a mock Google GenAI server, with the OpenInference Google ADK instrumentation.

NOTE: Google ADK imports are deferred inside run_agent() (in common.py) so that
instrumentation is fully active before ADK caches its tracer at import time.
"""

from common import run, run_agent


def instrument():
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    from openinference.instrumentation.config import TraceConfig

    GoogleADKInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))
    print("  [instrument] GoogleADKInstrumentor activated")


if __name__ == "__main__":
    run(
        "OpenInference: Google ADK Conformance Test",
        instrument,
        [run_agent],
    )
