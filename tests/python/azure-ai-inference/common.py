"""Shared test infrastructure for Azure AI Inference conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def run_chat(client):
    """Scenario: basic chat completion."""
    from azure.ai.inference.models import UserMessage

    print("  [chat] basic chat completion")
    resp = client.complete(
        model="gpt-4o-mini",
        messages=[UserMessage(content="Say hello.")],
    )
    print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming(client):
    """Scenario: streaming chat completion."""
    from azure.ai.inference.models import UserMessage

    print("  [chat_streaming] streaming chat completion")
    stream = client.complete(
        model="gpt-4o-mini",
        messages=[UserMessage(content="Tell me a joke.")],
        stream=True,
    )
    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    print(f"    -> {text[:60]}")


def run_embeddings(client):
    """Scenario: embedding generation."""
    print("  [embeddings] embedding generation")
    resp = client.embed(
        model="text-embedding-3-small",
        input=["Hello, world!"],
    )
    print(f"    -> embedding dim: {len(resp.data[0].embedding)}")


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()

    # Enable Azure SDK tracing via environment variable
    import os
    os.environ.setdefault("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")

    # Explicitly set the azure-core tracing implementation to OpenTelemetry
    from azure.core.settings import settings as azure_settings
    azure_settings.tracing_implementation = "opentelemetry"

    instrument_fn()

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

    for scenario in scenarios:
        if scenario == run_embeddings:
            scenario(embed_client)
        else:
            scenario(chat_client)

    flush_and_shutdown(tp, lp, mp)
