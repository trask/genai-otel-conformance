"""Conformance test: OpenInference (Arize) AutoGen instrumentation.

Exercises: agent_run
against a mock OpenAI server, with the OpenInference AutoGen instrumentation.
Uses pyautogen<1.0 (old API) which is compatible with openinference-instrumentation-autogen.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from openinference.instrumentation.autogen import AutogenInstrumentor
    AutogenInstrumentor().instrument()


def run_agent():
    """Scenario: basic agent execution via pyautogen."""
    import autogen

    print("  [agent_run] basic AutoGen agent execution")

    config_list = [
        {"model": "gpt-4o-mini", "api_key": "mock-key", "base_url": MOCK_BASE_URL}
    ]

    assistant = autogen.ConversableAgent(
        "assistant",
        llm_config={"config_list": config_list},
    )

    reply = assistant.generate_reply(
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(f"    -> {str(reply)[:60]}")


def main():
    print("=== OpenInference: AutoGen Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    run_agent()

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
