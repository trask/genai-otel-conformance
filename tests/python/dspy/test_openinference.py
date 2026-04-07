"""Conformance test: OpenInference (Arize) DSPy instrumentation.

Exercises: chat via DSPy Predict module
against a mock OpenAI server, with the OpenInference DSPy instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from openinference.instrumentation.dspy import DSPyInstrumentor
    DSPyInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


def run_chat():
    print("  [chat] basic chat via DSPy LM")
    import dspy

    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        cache=False,
    )
    dspy.configure(lm=lm)

    # Use raw LM call to avoid DSPy's JSON parsing requirement
    result = lm("Say hello.")
    print(f"    -> {str(result)[:60]}")


def run_chat_tool_call():
    print("  [chat_tool_call] tool calling via DSPy LM")
    import dspy

    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        cache=False,
    )
    dspy.configure(lm=lm)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    result = lm(
        messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
        tools=tools,
    )
    print(f"    -> {str(result)[:60]}")


def main():
    print("=== OpenInference: DSPy Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    run_chat()
    run_chat_tool_call()

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)
    print("Done.")


if __name__ == "__main__":
    main()
