"""Conformance test: OpenInference (Arize) Google GenAI instrumentation.

Uses openinference-instrumentation-google-genai to instrument the
google-genai SDK.  Exercises: chat, chat_streaming.
"""

import os

from opentelemetry import trace

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def instrument():
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    GoogleGenAIInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


def run_chat():
    """Scenario: basic chat completion via Google GenAI."""
    from google import genai
    from google.genai import types

    print("  [chat] basic chat completion via Google GenAI")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say hello.",
    )
    print(f"    -> {response.text[:60]}")


def run_chat_streaming():
    """Scenario: streaming chat completion via Google GenAI."""
    from google import genai
    from google.genai import types

    print("  [chat_streaming] streaming chat via Google GenAI")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    text = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents="Tell me a joke.",
    ):
        if chunk.text:
            text += chunk.text
    print(f"    -> {text[:60]}")


def run_chat_tool_call():
    """Scenario: chat with tool calling via Google GenAI."""
    from google import genai
    from google.genai import types

    print("  [chat_tool_call] chat with tool calling via Google GenAI")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "location": types.Schema(type="STRING", description="City name"),
                },
                required=["location"],
            ),
        )
    ])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What's the weather in Seattle?",
        config=types.GenerateContentConfig(tools=[tool]),
    )
    if response.candidates and response.candidates[0].content.parts:
        part = response.candidates[0].content.parts[0]
        if hasattr(part, "function_call") and part.function_call:
            print(f"    -> tool_call: {part.function_call.name}")
        else:
            print(f"    -> {response.text[:60]}")
    else:
        print(f"    -> {response.text[:60]}")


def run_embeddings():
    """Scenario: embedding generation via Google GenAI."""
    from google import genai
    from google.genai import types

    print("  [embeddings] embedding generation via Google GenAI")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    response = client.models.embed_content(
        model="text-embedding-004",
        contents="Hello, world!",
    )
    print(f"    -> embedding dim: {len(response.embeddings[0].values)}")


def main():
    print("=== OpenInference: Google GenAI Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    run_chat()
    run_chat_streaming()
    run_chat_tool_call()
    run_embeddings()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
