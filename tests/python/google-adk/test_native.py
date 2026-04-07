"""Conformance test: Google ADK native OTel instrumentation.

Google ADK automatically emits OTel spans when a global tracer provider is set.
Exercises: agent_run.

NOTE: Google imports are deferred to run_agent() so the tracer provider is
already active before ADK caches its tracer at import time.
"""

from common import run, run_agent


def instrument():
    """No-op: ADK native tracing is automatically enabled by the tracer provider."""
    pass


if __name__ == "__main__":
    run(
        "Native: Google ADK Conformance Test",
        instrument,
        [run_agent],
    )
