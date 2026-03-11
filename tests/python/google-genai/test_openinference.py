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


def main():
    print("=== OpenInference: Google GenAI Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()

    run_chat()
    run_chat_streaming()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
