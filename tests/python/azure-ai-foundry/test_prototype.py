"""Conformance test: Azure AI Foundry memory operations with prototype instrumentation.

Exercises: create_memory_store, update_memory, search_memory,
           delete_memory (scope), delete_memory_store
against a mock Azure AI Foundry server, with prototype OTel spans.
"""

from common import run, run_memory_operations


def instrument():
    """No-op: prototype instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Prototype: Azure AI Foundry Memory Conformance Test",
        instrument,
        [run_memory_operations],
    )
