"""Conformance test: OpenInference (Arize) Bedrock instrumentation.

Exercises: converse
against a mock Bedrock server, with the OpenInference Bedrock instrumentation.
"""

from common import run, run_converse, run_converse_tool_call, run_embeddings


def instrument():
    from openinference.instrumentation.bedrock import BedrockInstrumentor
    from openinference.instrumentation.config import TraceConfig

    BedrockInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run(
        "OpenInference: AWS Bedrock Conformance Test",
        instrument,
        [run_converse, run_embeddings],
    )
