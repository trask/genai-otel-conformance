"""Conformance test: prototype instrumentation for Google GenAI.

Exercises: chat, chat_streaming, embeddings
against a mock Google GenAI server, with manual OTel spans.
"""

import json
import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat completion with prototype instrumentation."""
    from google import genai
    from google.genai import types

    print("  [chat] basic chat completion via Google GenAI (prototype)")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    request_model = "gemini-2.0-flash"
    with _prototype_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "google_genai")
        span.set_attribute("gen_ai.request.model", request_model)
        prompt_text = "Say hello."
        response = client.models.generate_content(
            model=request_model,
            contents=prompt_text,
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
            if hasattr(response.usage_metadata, "thoughts_token_count") and response.usage_metadata.thoughts_token_count:
                span.set_attribute("gen_ai.usage.reasoning.output_tokens", response.usage_metadata.thoughts_token_count)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.input.messages": json.dumps([
                {"role": "user", "parts": [{"type": "text", "content": prompt_text}]}
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": response.text}],
                    "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
                }
            ]),
        }
        if response.model_version:
            event_attrs["gen_ai.response.model"] = response.model_version
        if response.candidates and response.candidates[0].finish_reason:
            event_attrs["gen_ai.response.finish_reasons"] = [str(response.candidates[0].finish_reason)]
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            if hasattr(response.usage_metadata, "prompt_token_count") and response.usage_metadata.prompt_token_count:
                event_attrs["gen_ai.usage.input_tokens"] = response.usage_metadata.prompt_token_count
            if hasattr(response.usage_metadata, "candidates_token_count") and response.usage_metadata.candidates_token_count:
                event_attrs["gen_ai.usage.output_tokens"] = response.usage_metadata.candidates_token_count
            if hasattr(response.usage_metadata, "thoughts_token_count") and response.usage_metadata.thoughts_token_count:
                event_attrs["gen_ai.usage.reasoning.output_tokens"] = response.usage_metadata.thoughts_token_count
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {response.text[:60]}")


def run_chat_tool_call():
    """Scenario: chat with tool calling with prototype instrumentation."""
    from google import genai
    from google.genai import types

    print("  [chat_tool_call] chat with tool calling via Google GenAI (prototype)")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    request_model = "gemini-2.0-flash"
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
    tools = [tool]
    with _prototype_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "google_genai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([{
            "function_declarations": [{
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            }]
        }]))
        response = client.models.generate_content(
            model=request_model,
            contents="What's the weather in Seattle?",
            config=types.GenerateContentConfig(tools=tools),
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
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if hasattr(part, "function_call") and part.function_call:
                print(f"    -> tool_call: {part.function_call.name}")
            else:
                print(f"    -> {response.text[:60]}")
        else:
            print(f"    -> {response.text[:60]}")


def run_chat_streaming():
    """Scenario: streaming chat completion with prototype instrumentation."""
    from google import genai
    from google.genai import types

    print("  [chat_streaming] streaming chat via Google GenAI (prototype)")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    request_model = "gemini-2.0-flash"
    with _prototype_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
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
            if hasattr(last_chunk.usage_metadata, "thoughts_token_count") and last_chunk.usage_metadata.thoughts_token_count:
                span.set_attribute("gen_ai.usage.reasoning.output_tokens", last_chunk.usage_metadata.thoughts_token_count)
        print(f"    -> {text[:60]}")


def run_embeddings():
    """Scenario: embedding generation with prototype instrumentation."""
    from google import genai
    from google.genai import types

    print("  [embeddings] embedding generation via Google GenAI (prototype)")
    client = genai.Client(
        api_key="mock-key",
        http_options=types.HttpOptions(
            base_url=MOCK_BASE_URL,
            api_version="v1beta",
        ),
    )
    request_model = "text-embedding-004"
    with _prototype_tracer.start_as_current_span("embeddings text-embedding-004") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "google_genai")
        span.set_attribute("gen_ai.request.model", request_model)
        response = client.models.embed_content(
            model=request_model,
            contents="Hello, world!",
        )
        print(f"    -> embedding dim: {len(response.embeddings[0].values)}")


def main():
    print("=== Prototype: Google GenAI Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    run_chat()
    run_chat_tool_call()
    run_chat_streaming()
    run_embeddings()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
