"""Conformance test: OpenInference (Arize) Bedrock instrumentation.

Exercises: converse
against a mock Bedrock server, with the OpenInference Bedrock instrumentation.
"""

from opentelemetry import trace

from common import run, run_converse, run_embeddings


def instrument():
    from openinference.instrumentation.bedrock import BedrockInstrumentor
    BedrockInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run(
        "OpenInference: AWS Bedrock Conformance Test",
        instrument,
        [run_converse, run_embeddings],
    )
