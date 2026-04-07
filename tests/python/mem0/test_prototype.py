"""Conformance test: Mem0 prototype memory instrumentation.

Exercises: add memory, search memory, delete memory
against a mock Mem0 server, with prototype span instrumentation.
"""

from common import (
    run,
    run_add_memory,
    run_search_memory,
    run_delete_memory,
    run_delete_all_memories,
)


def instrument():
    """No-op: prototype instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Prototype: Mem0 Memory Conformance Test",
        instrument,
        [run_add_memory, run_search_memory, run_delete_memory, run_delete_all_memories],
    )
