"""Shared test infrastructure for Anthropic conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def run_chat(client):
    """Scenario: basic message."""
    print("  [chat] basic message")
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(f"    -> {resp.content[0].text[:60]}")


def run_chat_streaming(client):
    """Scenario: streaming message."""
    print("  [chat_streaming] streaming message")
    text = ""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "Tell me a joke."}],
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk
    print(f"    -> {text[:60]}")


def run(title, instrument_fn):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    import anthropic
    client = anthropic.Anthropic(base_url=MOCK_BASE_URL, api_key="mock-key")

    run_chat(client)
    run_chat_streaming(client)

    flush_and_shutdown(tp, lp, mp)
