"""Conformance test: OpenInference (Arize) LlamaIndex instrumentation.

Exercises: chat
against a mock OpenAI server, with the OpenInference LlamaIndex instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


def run_chat(llm):
    print("  [chat] basic chat completion")
    from llama_index.core.llms import ChatMessage, MessageRole
    resp = llm.chat([ChatMessage(role=MessageRole.USER, content="Say hello.")])
    print(f"    -> {str(resp)[:60]}")


def run_chat_streaming(llm):
    print("  [chat_streaming] streaming chat completion")
    from llama_index.core.llms import ChatMessage, MessageRole
    text = ""
    stream_resp = llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Tell me a joke.")])
    for token in stream_resp:
        text += token.delta
    print(f"    -> {text[:60]}")


def main():
    print("=== OpenInference: LlamaIndex Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    llm = LlamaOpenAI(
        model="gpt-4o-mini",
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )

    run_chat(llm)
    run_chat_streaming(llm)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
