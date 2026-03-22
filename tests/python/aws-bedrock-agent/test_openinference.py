"""Conformance test: OpenInference (Arize) Bedrock instrumentation for Agent Runtime.

Exercises: invoke_agent
against a mock Bedrock Agent server, with the OpenInference Bedrock instrumentation.
"""

from opentelemetry import trace

from common import run, run_invoke_agent


def instrument():
    from openinference.instrumentation.bedrock import BedrockInstrumentor
    BedrockInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run(
        "OpenInference: AWS Bedrock Agent Conformance Test",
        instrument,
        [run_invoke_agent],
    )
