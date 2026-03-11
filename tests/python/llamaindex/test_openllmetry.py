"""Conformance test: OpenLLMetry (Traceloop) LlamaIndex instrumentation.

Exercises: chat, chat_streaming, agent
against a mock OpenAI server, with the Traceloop LlamaIndex instrumentation.
"""

from common import run


def instrument():
    from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    LlamaIndexInstrumentor().instrument()
    OpenAIInstrumentor().instrument()


if __name__ == "__main__":
    run("OpenLLMetry: LlamaIndex Conformance Test", instrument)
