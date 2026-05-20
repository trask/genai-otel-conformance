"""Conformance test: OpenInference (Arize) Claude Agent SDK instrumentation.

Exercises: agent_query
against a mock Claude CLI, with the OpenInference Claude Agent SDK instrumentation.
"""

from common import run


def instrument():
    from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    from openinference.instrumentation.config import TraceConfig

    ClaudeAgentSDKInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run("OpenInference: Claude Agent SDK Conformance Test", instrument)
