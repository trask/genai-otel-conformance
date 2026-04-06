"""Conformance test: AWS Bedrock Agent invoke_agent with manual instrumentation.

Exercises: invoke_agent (Bedrock Agent Runtime InvokeAgent API)
against a mock Bedrock server, with manual span instrumentation.
"""

from common import run, run_invoke_agent


def instrument():
    """No-op: manual instrumentation only."""
    pass


if __name__ == "__main__":
    run(
        "Manual: AWS Bedrock Agent Invoke Agent Conformance Test",
        instrument,
        [run_invoke_agent],
    )
