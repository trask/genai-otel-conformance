"""Conformance test: Google ADK manual memory instrumentation.

Exercises: memory operations (add_session_to_memory, search_memory)
against a mock Google GenAI server, with manual span instrumentation.

NOTE: Google imports are deferred to run_memory_operations() so the tracer
provider is already active before ADK caches its tracer at import time.
"""

from common import run, run_memory_operations


def instrument():
    """No-op: manual instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Manual: Google ADK Memory Conformance Test",
        instrument,
        [run_memory_operations],
    )
