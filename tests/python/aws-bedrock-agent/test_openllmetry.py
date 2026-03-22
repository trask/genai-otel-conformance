"""Conformance test: OpenLLMetry (Traceloop) Bedrock instrumentation for Agent Runtime.

Exercises: invoke_agent
against a mock Bedrock Agent server, with the Traceloop Bedrock instrumentation.
"""

from common import run, run_invoke_agent


def instrument():
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
    BedrockInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OpenLLMetry: AWS Bedrock Agent Conformance Test",
        instrument,
        [run_invoke_agent],
    )
