"""Conformance test: OpenAI Assistants invoke_agent with manual instrumentation.

Exercises: invoke_agent (OpenAI Assistants API: create thread, run, poll)
against a mock OpenAI server, with manual span instrumentation.
"""

from common import run, run_invoke_agent


def instrument():
    """No-op: manual instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Manual: OpenAI Assistants Invoke Agent Conformance Test",
        instrument,
        [run_invoke_agent],
    )
