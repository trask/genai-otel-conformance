"""Conformance test: OpenInference (Arize) Groq instrumentation.

Exercises: chat, chat_streaming
against a mock OpenAI-compatible server, with the OpenInference Groq instrumentation.
"""

from opentelemetry import trace

from common import run, run_chat, run_chat_streaming, run_chat_tool_call


def instrument():
    from openinference.instrumentation.groq import GroqInstrumentor
    GroqInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run(
        "OpenInference: Groq Conformance Test",
        instrument,
        [run_chat, run_chat_streaming, run_chat_tool_call],
    )
