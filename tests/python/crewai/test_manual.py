"""Conformance test: CrewAI manual memory instrumentation.

Exercises: memory operations (remember, recall, forget, reset)
against a mock OpenAI server, with manual span instrumentation.
"""

from common import run, run_memory_operations


def instrument():
    """No-op: manual instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Manual: CrewAI Memory Conformance Test",
        instrument,
        [run_memory_operations],
    )
