"""Conformance test: OpenInference (Arize) OpenAI instrumentation."""

from opentelemetry import trace

from common import run, run_chat, run_chat_streaming, run_embeddings


def instrument():
    from openinference.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run(
        "OpenInference: OpenAI Conformance Test",
        instrument,
        [run_chat, run_chat_streaming, run_embeddings],
    )
