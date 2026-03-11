"""Conformance test: OpenLLMetry (Traceloop) Bedrock instrumentation.

Exercises: converse
against a mock Bedrock server, with the Traceloop Bedrock instrumentation.
"""

from common import run, run_converse


def instrument():
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
    BedrockInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OpenLLMetry: AWS Bedrock Conformance Test",
        instrument,
        [run_converse],
    )
