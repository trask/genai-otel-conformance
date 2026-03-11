"""Conformance test: OpenInference (Arize) Mistral AI instrumentation.

Exercises: chat
against a mock OpenAI-compatible server, with the OpenInference Mistral AI instrumentation.
Uses mistralai<1.0 (old API) which is compatible with openinference-instrumentation-mistralai.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def instrument():
    from openinference.instrumentation.mistralai import MistralAIInstrumentor
    MistralAIInstrumentor().instrument()


def run_chat(client):
    from mistralai.models.chat_completion import ChatMessage

    print("  [chat] basic chat completion")
    resp = client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(content="Say hello.", role="user")],
    )
    print(f"    -> {resp.choices[0].message.content[:60]}")


def main():
    print("=== OpenInference: Mistral AI Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    from mistralai.client import MistralClient
    client = MistralClient(api_key="mock-key", endpoint=MOCK_BASE_URL)

    run_chat(client)

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
