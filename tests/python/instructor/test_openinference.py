"""Conformance test: OpenInference (Arize) Instructor instrumentation.

Exercises: chat (structured extraction)
against a mock OpenAI server, with the OpenInference Instructor instrumentation.
"""

from common import run, run_chat, run_chat_tool_call


def instrument():
    from openinference.instrumentation.instructor import InstructorInstrumentor
    from openinference.instrumentation.config import TraceConfig

    InstructorInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run(
        "OpenInference: Instructor Conformance Test",
        instrument,
        [run_chat, run_chat_tool_call],
    )
