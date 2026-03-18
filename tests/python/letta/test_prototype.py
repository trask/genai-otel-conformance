"""Conformance test: Letta prototype memory instrumentation.

Exercises: update memory (core memory block), search archival memory,
delete archival memory against a mock Letta server, with prototype span
instrumentation.
"""

from common import run, run_update_memory, run_search_memory, run_delete_memory


def instrument():
    """No-op: prototype instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Prototype: Letta Memory Conformance Test",
        instrument,
        [run_update_memory, run_search_memory, run_delete_memory],
    )
