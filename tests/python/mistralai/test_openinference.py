"""Conformance test: OpenInference (Arize) Mistral AI instrumentation.

Exercises: chat
against a mock OpenAI-compatible server, with the Arize Mistral AI instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def instrument():
    from openinference.instrumentation.mistralai import MistralAIInstrumentor
    from openinference.instrumentation.config import TraceConfig

    MistralAIInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


def run_chat(client):
    print("  [chat] basic chat completion")
    resp = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming(client):
    print("  [chat_streaming] streaming chat completion")
    text = ""
    stream = client.chat.stream(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Tell me a joke."}],
    )
    for event in stream:
        if event.data.choices and event.data.choices[0].delta.content:
            text += event.data.choices[0].delta.content
    print(f"    -> {text[:60]}")


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
    resp = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
        tools=tools,
    )
    choice = resp.choices[0]
    if choice.message.tool_calls:
        print(f"    -> tool_call: {choice.message.tool_calls[0].function.name}")
    else:
        print(f"    -> {choice.message.content[:60]}")


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

    from mistralai.client import Mistral
    client = Mistral(
        api_key="mock-key",
        server_url=MOCK_BASE_URL,
    )

    run_chat(client)
    try:
        run_chat_streaming(client)
    except Exception as e:
        print(f"    WARNING: streaming failed: {e}")

    run_chat_tool_call(client)
    run_embeddings(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
