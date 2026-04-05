"""Conformance test: prototype instrumentation for Instructor."""

import json
import os

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat_prototype(client):
    """Scenario: structured extraction via Instructor with prototype instrumentation."""
    from pydantic import BaseModel

    print("  [chat] structured extraction via Instructor (prototype)")
    request_model = "gpt-4o-mini"

    class Greeting(BaseModel):
        message: str

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        messages = [{"role": "user", "content": "Say hello."}]
        resp, completion = client.chat.completions.create_with_completion(
            model=request_model,
            messages=messages,
            response_model=Greeting,
        )
        span.set_attribute("gen_ai.response.model", completion.model)
        span.set_attribute("gen_ai.response.id", completion.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in completion.choices])
        if completion.usage:
            span.set_attribute("gen_ai.usage.input_tokens", completion.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion.usage.completion_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.response.id": completion.id,
            "gen_ai.response.model": completion.model,
            "gen_ai.response.finish_reasons": [c.finish_reason for c in completion.choices],
            "gen_ai.input.messages": json.dumps([
                {"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]}
                for m in messages
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": c.message.role,
                    "parts": [{"type": "text", "content": c.message.content}],
                    "finish_reason": c.finish_reason,
                }
                for c in completion.choices
            ]),
        }
        if completion.usage:
            event_attrs["gen_ai.usage.input_tokens"] = completion.usage.prompt_tokens
            event_attrs["gen_ai.usage.output_tokens"] = completion.usage.completion_tokens
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {resp.message[:60]}")


def main():
    print("=== Prototype: Instructor Conformance Test ===")

    tp, lp, mp = setup_otel()

    import openai
    from instructor.core.client import from_openai

    client = from_openai(
        openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key"),
    )

    run_chat_prototype(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
