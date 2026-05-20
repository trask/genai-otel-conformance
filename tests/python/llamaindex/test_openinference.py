"""Conformance test: OpenInference (Arize) LlamaIndex instrumentation.

Exercises: chat, chat_streaming, agent
against a mock OpenAI server, with the OpenInference LlamaIndex instrumentation.
"""

from common import run


def instrument():
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.config import TraceConfig

    LlamaIndexInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run("OpenInference: LlamaIndex Conformance Test", instrument)
