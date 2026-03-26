"""Conformance test: reference instrumentation for Google GenAI.

Exercises: chat, chat_streaming, embeddings
against a mock Google GenAI server, with manual OTel spans.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_chat():
    """Scenario: basic chat completion with reference instrumentation."""
    from google import genai
    from google.genai import types

    print("  [chat] basic chat completion via Google GenAI (reference)")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    request_model = "gemini-2.0-flash"
    with _reference_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "google_genai")
        span.set_attribute("gen_ai.request.model", request_model)
        response = client.models.generate_content(
            model=request_model,
            contents="Say hello.",
        )
        if response.model_version:
            span.set_attribute("gen_ai.response.model", response.model_version)
        if response.candidates and response.candidates[0].finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [str(response.candidates[0].finish_reason)])
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            if hasattr(response.usage_metadata, "prompt_token_count") and response.usage_metadata.prompt_token_count:
                span.set_attribute("gen_ai.usage.input_tokens", response.usage_metadata.prompt_token_count)
            if hasattr(response.usage_metadata, "candidates_token_count") and response.usage_metadata.candidates_token_count:
                span.set_attribute("gen_ai.usage.output_tokens", response.usage_metadata.candidates_token_count)
        print(f"    -> {response.text[:60]}")


def run_chat_streaming():
    """Scenario: streaming chat completion with reference instrumentation."""
    from google import genai
    from google.genai import types

    print("  [chat_streaming] streaming chat via Google GenAI (reference)")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    request_model = "gemini-2.0-flash"
    with _reference_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "google_genai")
        span.set_attribute("gen_ai.request.model", request_model)
        text = ""
        last_chunk = None
        for chunk in client.models.generate_content_stream(
            model=request_model,
            contents="Tell me a joke.",
        ):
            if chunk.text:
                text += chunk.text
            last_chunk = chunk
        if last_chunk and last_chunk.model_version:
            span.set_attribute("gen_ai.response.model", last_chunk.model_version)
        if last_chunk and last_chunk.candidates and last_chunk.candidates[0].finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [str(last_chunk.candidates[0].finish_reason)])
        if last_chunk and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
            if hasattr(last_chunk.usage_metadata, "prompt_token_count") and last_chunk.usage_metadata.prompt_token_count:
                span.set_attribute("gen_ai.usage.input_tokens", last_chunk.usage_metadata.prompt_token_count)
            if hasattr(last_chunk.usage_metadata, "candidates_token_count") and last_chunk.usage_metadata.candidates_token_count:
                span.set_attribute("gen_ai.usage.output_tokens", last_chunk.usage_metadata.candidates_token_count)
        print(f"    -> {text[:60]}")


def run_embeddings():
    """Scenario: embedding generation with reference instrumentation."""
    from google import genai
    from google.genai import types

    print("  [embeddings] embedding generation via Google GenAI (reference)")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    request_model = "text-embedding-004"
    with _reference_tracer.start_as_current_span("embeddings text-embedding-004") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "google_genai")
        span.set_attribute("gen_ai.request.model", request_model)
        response = client.models.embed_content(
            model=request_model,
            contents="Hello, world!",
        )
        print(f"    -> embedding dim: {len(response.embeddings[0].values)}")


def main():
    print("=== Reference: Google GenAI Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – reference instrumentation only

    run_chat()
    run_chat_streaming()
    run_embeddings()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
