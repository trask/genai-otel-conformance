"""Conformance test: OpenInference (Arize) LlamaIndex instrumentation.

Exercises: chat, chat_streaming, agent
against a mock OpenAI server, with the OpenInference LlamaIndex instrumentation.
"""

from opentelemetry import trace

from common import run


def instrument():
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    LlamaIndexInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run("OpenInference: LlamaIndex Conformance Test", instrument)
