"""Conformance test: OpenInference (Arize) Groq instrumentation.

Exercises: chat, chat_streaming
against a mock OpenAI-compatible server, with the OpenInference Groq instrumentation.
"""

from common import run, run_chat, run_chat_streaming, run_chat_tool_call


def instrument():
    from openinference.instrumentation.groq import GroqInstrumentor
    from openinference.instrumentation.config import TraceConfig

    GroqInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run(
        "OpenInference: Groq Conformance Test",
        instrument,
        [run_chat, run_chat_streaming, run_chat_tool_call],
    )
