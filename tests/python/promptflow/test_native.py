"""Conformance test: Promptflow native OTel instrumentation.

Exercises: chat via @trace-decorated function
against a mock OpenAI server, with Promptflow's built-in tracing.
"""

import os

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def run_chat():
    print("  [chat] basic chat via @trace decorator")
    from promptflow.tracing import trace as pf_trace
    import openai

    client = openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")

    @pf_trace
    def chat_completion(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    result = chat_completion("Say hello.")
    print(f"    -> {result[:60]}")


def main():
    print("=== Native: Promptflow Conformance Test ===")

    # Promptflow disables tracing by default (PF_DISABLE_TRACING defaults to "true")
    os.environ["PF_DISABLE_TRACING"] = "false"

    # start_trace() creates its own TracerProvider, so call it FIRST,
    # then add our OTLP exporter to the provider it created.
    from promptflow.tracing import start_trace
    start_trace()

    # Reuse the shared provider setup while preserving Promptflow's tracer provider.
    tp, lp, mp = setup_otel(trace.get_tracer_provider())

    run_chat()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
