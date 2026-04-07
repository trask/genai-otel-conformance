"""Conformance test: LiteLLM native OTel instrumentation.

LiteLLM has built-in OTel support via `litellm.callbacks = ["otel"]`.
Exercises: chat, chat_streaming, embeddings.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    import litellm
    litellm.callbacks = ["otel"]


def run_chat():
    import litellm
    print("  [chat] basic chat completion via LiteLLM")
    resp = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello."}],
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )
    print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming():
    import litellm
    print("  [chat_streaming] streaming chat via LiteLLM")
    resp = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        stream=True,
    )
    text = ""
    for chunk in resp:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    print(f"    -> {text[:60]}")


def run_chat_tool_call():
    import litellm
    print("  [chat_tool_call] chat with tool calling via LiteLLM")
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
    resp = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        tools=tools,
    )
    choice = resp.choices[0]
    if choice.message.tool_calls:
        print(f"    -> tool_call: {choice.message.tool_calls[0].function.name}")
    else:
        print(f"    -> {choice.message.content[:60]}")


def run_embeddings():
    import litellm
    print("  [embeddings] embedding generation via LiteLLM")
    resp = litellm.embedding(
        model="openai/text-embedding-3-small",
        input=["Hello, world!"],
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )
    print(f"    -> embedding dim: {len(resp.data[0]['embedding'])}")


def main():
    print("=== Native: LiteLLM Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    run_chat()
    run_chat_streaming()
    run_chat_tool_call()
    run_embeddings()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
