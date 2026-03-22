"""Conformance test: OTel contrib opentelemetry-instrumentation-openai-v2.

Exercises: invoke_agent (Assistants API: create assistant, thread, run)
against a mock server.
"""

from common import run, run_invoke_agent


def instrument():
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
    OpenAIInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OTel Contrib: OpenAI Assistants Conformance Test",
        instrument,
        [run_invoke_agent],
    )
