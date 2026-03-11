"""Shared test infrastructure for Instructor conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def run_chat(client):
    """Scenario: basic structured extraction via Instructor."""
    from pydantic import BaseModel

    print("  [chat] structured extraction via Instructor")

    class Greeting(BaseModel):
        message: str

    # Instructor patches the OpenAI client to extract structured output.
    # With the mock server, it will parse the response into the model.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello."}],
        response_model=Greeting,
    )
    print(f"    -> {resp.message[:60]}")


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    import openai
    import instructor

    client = instructor.from_openai(
        openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key"),
    )

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
