"""Conformance test: OpenInference (Arize) Mistral AI instrumentation.

Exercises: chat
against a mock OpenAI-compatible server, with the OpenInference Mistral AI instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def instrument():
    from openinference.instrumentation.mistralai import MistralAIInstrumentor
    MistralAIInstrumentor().instrument()


def run_chat(client):
    print("  [chat] basic chat completion")
    resp = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(f"    -> {resp.choices[0].message.content[:60]}")


def run_embeddings(client):
    print("  [embeddings] embedding generation")
    resp = client.embeddings.create(
        model="mistral-embed",
        inputs=["Hello, world!"],
    )
    print(f"    -> embedding dim: {len(resp.data[0].embedding)}")


def main():
    print("=== OpenInference: Mistral AI Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    from mistralai import Mistral
    client = Mistral(api_key="mock-key", server_url=MOCK_BASE_URL)

    run_chat(client)

    run_embeddings(client)

    import time
    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
