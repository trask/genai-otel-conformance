"""Conformance test: OpenLLMetry (Traceloop) OpenAI instrumentation."""

from common import run, run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings


def instrument():
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OpenLLMetry: OpenAI Conformance Test",
        instrument,
        [run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings],
    )
