"""Conformance test: OpenLLMetry (Traceloop) Cohere instrumentation.

Exercises: chat
against a mock Cohere server, with the Traceloop Cohere instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def instrument():
    from opentelemetry.instrumentation.cohere import CohereInstrumentor
    CohereInstrumentor().instrument()


def run_chat(client):
    print("  [chat] basic chat completion")
    resp = client.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(f"    -> {resp.message.content[0].text[:60]}")


def main():
    print("=== OpenLLMetry: Cohere Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    import cohere
    client = cohere.ClientV2(
        api_key="mock-key",
        base_url=MOCK_BASE_URL,
    )

    run_chat(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
