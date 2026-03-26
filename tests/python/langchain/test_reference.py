"""Conformance test: reference instrumentation for LangChain."""

import os

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = trace.get_tracer("gen_ai.reference")


def _usage_value(usage, key):
    if not usage:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def run_chat_reference(llm, request_model):
    """Scenario: basic chat completion with reference instrumentation."""
    print("  [chat] basic chat completion (reference)")
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = llm.invoke("Say hello.")
        meta = getattr(resp, "response_metadata", {})
        if meta.get("model_name"):
            span.set_attribute("gen_ai.response.model", meta["model_name"])
        if getattr(resp, "id", None):
            span.set_attribute("gen_ai.response.id", resp.id)
        if meta.get("finish_reason"):
            span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
        usage = getattr(resp, "usage_metadata", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        print(f"    -> {resp.content[:60]}")


def run_chat_streaming_reference(llm, request_model):
    """Scenario: streaming chat completion with reference instrumentation."""
    print("  [chat_streaming] streaming chat completion (reference)")
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        text = ""
        full = None
        for chunk in llm.stream("Tell me a joke."):
            text += chunk.content
            full = chunk if full is None else full + chunk
        if full:
            meta = getattr(full, "response_metadata", {})
            if meta.get("model_name"):
                span.set_attribute("gen_ai.response.model", meta["model_name"])
            if getattr(full, "id", None):
                span.set_attribute("gen_ai.response.id", full.id)
            if meta.get("finish_reason"):
                span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
            usage = getattr(full, "usage_metadata", None)
            input_tokens = _usage_value(usage, "input_tokens")
            output_tokens = _usage_value(usage, "output_tokens")
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        print(f"    -> {text[:60]}")


def run_embeddings_reference():
    """Scenario: embedding generation with reference instrumentation."""
    print("  [embeddings] embedding generation (reference)")
    from langchain_openai import OpenAIEmbeddings

    request_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(
        model=request_model,
        openai_api_base=MOCK_BASE_URL,
        openai_api_key="mock-key",
    )
    with _reference_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        result = embeddings.embed_query("Hello, world!")
        print(f"    -> embedding dim: {len(result)}")


def main():
    print("=== Reference: LangChain Conformance Test ===")

    tp, lp, mp = setup_otel()

    from langchain_openai import ChatOpenAI

    request_model = "gpt-4o-mini"
    llm = ChatOpenAI(
        model=request_model,
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )

    run_chat_reference(llm, request_model)
    run_chat_streaming_reference(llm, request_model)
    run_embeddings_reference()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
