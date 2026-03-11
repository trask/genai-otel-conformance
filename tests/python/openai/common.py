"""Shared test infrastructure for OpenAI conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def run_chat(client):
    """Scenario: basic chat completion."""
    print("  [chat] basic chat completion")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming(client):
    """Scenario: streaming chat completion."""
    print("  [chat_streaming] streaming chat completion")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    print(f"    -> {text[:60]}")


def run_chat_tool_call(client):
    """Scenario: chat with tool calling."""
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
        tools=tools,
    )
    choice = resp.choices[0]
    if choice.message.tool_calls:
        print(f"    -> tool_call: {choice.message.tool_calls[0].function.name}")
    else:
        print(f"    -> {choice.message.content[:60]}")


def run_embeddings(client):
    """Scenario: embedding generation."""
    print("  [embeddings] embedding generation")
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello, world!",
    )
    print(f"    -> embedding dim: {len(resp.data[0].embedding)}")


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    import openai
    client = openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
