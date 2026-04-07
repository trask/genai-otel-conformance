"""Conformance test: CrewAI prototype memory instrumentation.

Exercises: memory operations (remember, recall, forget)
against a mock OpenAI server, with prototype span instrumentation.
"""

from common import run, run_memory_operations


def instrument():
    """No-op: prototype instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Prototype: CrewAI Memory Conformance Test",
        instrument,
        [run_memory_operations],
    )
