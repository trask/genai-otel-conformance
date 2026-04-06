"""Conformance test: Azure AI Foundry invoke_agent with manual instrumentation.

Exercises: invoke_agent (Azure AI Foundry Agents API: create agent, create
thread, run, poll) against a mock Azure AI Foundry server, with manual span
instrumentation.
"""

from common import run, run_invoke_agent


def instrument():
    """No-op: manual instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Manual: Azure AI Foundry Invoke Agent Conformance Test",
        instrument,
        [run_invoke_agent],
    )
