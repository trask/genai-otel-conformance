"""Conformance test: OpenLLMetry (Traceloop) Groq instrumentation."""

from common import run, run_chat, run_chat_streaming


def instrument():
    from opentelemetry.instrumentation.groq import GroqInstrumentor
    GroqInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OpenLLMetry: Groq Conformance Test",
        instrument,
        [run_chat, run_chat_streaming],
    )
