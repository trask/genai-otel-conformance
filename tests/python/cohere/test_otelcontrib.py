"""Conformance test: OTel contrib Cohere instrumentation.

Exercises: chat
against a mock Cohere server, with the OTel contrib Cohere instrumentation.
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


def run_chat_tool_call(client):
    print("  [chat_tool_call] chat with tool calling")
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
    resp = client.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
        tools=tools,
    )
    if hasattr(resp.message, "tool_calls") and resp.message.tool_calls:
        print(f"    -> tool_call: {resp.message.tool_calls[0].function.name}")
    else:
        print(f"    -> {resp.message.content[0].text[:60]}")


def run_embeddings(client):
    print("  [embeddings] embedding generation")
    resp = client.embed(
        model="embed-v4.0",
        texts=["Hello, world!"],
        input_type="search_document",
        embedding_types=["float"],
    )
    print(f"    -> embedding dim: {len(resp.embeddings.float_[0])}")


def main():
    print("=== OTel Contrib: Cohere Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    import cohere
    client = cohere.ClientV2(
        api_key="mock-key",
        base_url=MOCK_BASE_URL,
    )

    run_chat(client)
    run_chat_tool_call(client)
    run_embeddings(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
