"""Conformance test: reference instrumentation for Instructor."""

import os

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_chat_reference(client):
    """Scenario: structured extraction via Instructor with reference instrumentation."""
    from pydantic import BaseModel

    print("  [chat] structured extraction via Instructor (reference)")
    request_model = "gpt-4o-mini"

    class Greeting(BaseModel):
        message: str

    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        resp, completion = client.chat.completions.create_with_completion(
            model=request_model,
            messages=[{"role": "user", "content": "Say hello."}],
            response_model=Greeting,
        )
        span.set_attribute("gen_ai.response.model", completion.model)
        span.set_attribute("gen_ai.response.id", completion.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in completion.choices])
        if completion.usage:
            span.set_attribute("gen_ai.usage.input_tokens", completion.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion.usage.completion_tokens)
        print(f"    -> {resp.message[:60]}")


def main():
    print("=== Reference: Instructor Conformance Test ===")

    tp, lp, mp = setup_otel()

    import openai
    from instructor.core.client import from_openai

    client = from_openai(
        openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key"),
    )

    run_chat_reference(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
