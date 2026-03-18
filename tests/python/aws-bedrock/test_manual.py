"""Conformance test: AWS Bedrock manual memory instrumentation.

Exercises: memory operations (create_memory_store, update_memory,
search_memory, delete_memory, delete_memory_store) against a mock
Bedrock AgentCore server, with manual span instrumentation.
"""

from common import run, run_memory_operations


def instrument():
    """No-op: manual instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Manual: AWS Bedrock Memory Conformance Test",
        instrument,
        [run_memory_operations],
    )
