"""Conformance test: prototype instrumentation for Azure AI Inference."""

import os
from urllib.parse import urlparse

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat_prototype(client):
    """Scenario: basic chat completion with prototype instrumentation."""
    from azure.ai.inference.models import UserMessage

    print("  [chat] basic chat completion (prototype)")
    request_model = "gpt-4o-mini"
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "az.ai.inference")
        span.set_attribute("gen_ai.request.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        resp = client.complete(
            model=request_model,
            messages=[UserMessage(content="Say hello.")],
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        finish_reasons = [str(c.finish_reason) for c in resp.choices if c.finish_reason]
        if finish_reasons:
            span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
        print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming_prototype(client):
    """Scenario: streaming chat completion with prototype instrumentation."""
    from azure.ai.inference.models import UserMessage

    print("  [chat_streaming] streaming chat completion (prototype)")
    request_model = "gpt-4o-mini"
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "az.ai.inference")
        span.set_attribute("gen_ai.request.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        stream = client.complete(
            model=request_model,
            messages=[UserMessage(content="Tell me a joke.")],
            stream=True,
        )
        text = ""
        model = None
        response_id = None
        finish_reasons = []
        for chunk in stream:
            model = model or getattr(chunk, "model", None)
            response_id = response_id or getattr(chunk, "id", None)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reasons.append(str(chunk.choices[0].finish_reason))
        if model:
            span.set_attribute("gen_ai.response.model", model)
        if response_id:
            span.set_attribute("gen_ai.response.id", response_id)
        if finish_reasons:
            span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        print(f"    -> {text[:60]}")


def run_embeddings_prototype(client):
    """Scenario: embedding generation with prototype instrumentation."""
    print("  [embeddings] embedding generation (prototype)")
    request_model = "text-embedding-3-small"
    with _prototype_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "az.ai.inference")
        span.set_attribute("gen_ai.request.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        resp = client.embed(
            model=request_model,
            input=["Hello, world!"],
        )
        if resp.model:
            span.set_attribute("gen_ai.response.model", resp.model)
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
        print(f"    -> embedding dim: {len(resp.data[0].embedding)}")


def main():
    print("=== Prototype: Azure AI Inference Conformance Test ===")

    tp, lp, mp = setup_otel()

    import os as _os
    _os.environ.setdefault("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")

    from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
    from azure.core.credentials import AzureKeyCredential

    chat_client = ChatCompletionsClient(
        endpoint=MOCK_BASE_URL,
        credential=AzureKeyCredential("mock-key"),
    )
    embed_client = EmbeddingsClient(
        endpoint=MOCK_BASE_URL,
        credential=AzureKeyCredential("mock-key"),
    )

    run_chat_prototype(chat_client)
    run_chat_streaming_prototype(chat_client)
    run_embeddings_prototype(embed_client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
