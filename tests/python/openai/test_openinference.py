"""Conformance test: OpenInference (Arize) OpenAI instrumentation."""

from common import run, run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings


def instrument():
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from openinference.instrumentation.config import TraceConfig

    OpenAIInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run(
        "OpenInference: OpenAI Conformance Test",
        instrument,
        [run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings],
    )
