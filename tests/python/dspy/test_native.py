"""Conformance test: DSPy native OTel instrumentation.

Exercises: chat via DSPy Predict module
against a mock OpenAI server, with LiteLLM's built-in OpenTelemetry callback
(DSPy delegates to LiteLLM for HTTP calls).
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


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


def main():
    print("=== Native: DSPy Conformance Test ===")

    tp, lp, mp = setup_otel()

    # DSPy uses LiteLLM internally; enable LiteLLM's built-in OTel callback
    import litellm
    litellm.success_callback = ["otel"]
    litellm.failure_callback = ["otel"]

    run_chat()

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)
    print("Done.")


if __name__ == "__main__":
    main()
