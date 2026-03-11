"""Conformance test: OpenInference (Arize) Claude Agent SDK instrumentation.

Exercises: agent_query
against a mock Claude CLI, with the OpenInference Claude Agent SDK instrumentation.
"""

from opentelemetry import trace

from common import run


def instrument():
    from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run("OpenInference: Claude Agent SDK Conformance Test", instrument)
