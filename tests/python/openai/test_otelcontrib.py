"""Conformance test: OTel contrib opentelemetry-instrumentation-openai-v2."""

from common import run, run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings


def instrument():
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OTel Contrib: OpenAI v2 Conformance Test",
        instrument,
        [run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings],
    )
