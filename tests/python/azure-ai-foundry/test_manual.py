"""Conformance test: Azure AI Foundry memory operations with manual instrumentation.

Exercises: create_memory_store, update_memory, search_memory,
           delete_memory (scope), delete_memory_store
against a mock Azure AI Foundry server, with manual OTel spans demonstrating
what an instrumentation library should capture.
"""

from common import run, run_memory_operations


def instrument():
    """No-op: manual instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Manual: Azure AI Foundry Memory Conformance Test",
        instrument,
        [run_memory_operations],
    )
