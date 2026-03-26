"""Conformance test: reference instrumentation for LlamaIndex."""

import os

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_chat_reference(llm, request_model, request_temperature):
    """Scenario: basic chat completion with reference instrumentation."""
    from llama_index.core.llms import ChatMessage, MessageRole

    print("  [chat] basic chat completion (reference)")
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.request.temperature", request_temperature)
        resp = llm.chat([ChatMessage(role=MessageRole.USER, content="Say hello.")])
        raw = getattr(resp, "raw", None)
        if raw:
            if getattr(raw, "model", None):
                span.set_attribute("gen_ai.response.model", raw.model)
            if getattr(raw, "id", None):
                span.set_attribute("gen_ai.response.id", raw.id)
            if getattr(raw, "choices", None):
                span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in raw.choices])
            if getattr(raw, "usage", None) and raw.usage:
                span.set_attribute("gen_ai.usage.input_tokens", raw.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", raw.usage.completion_tokens)
        print(f"    -> {str(resp)[:60]}")


def run_chat_streaming_reference(llm, request_model, request_temperature):
    """Scenario: streaming chat completion with reference instrumentation."""
    from llama_index.core.llms import ChatMessage, MessageRole

    print("  [chat_streaming] streaming chat completion (reference)")
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.request.temperature", request_temperature)
        text = ""
        stream_resp = llm.stream_chat(
            [ChatMessage(role=MessageRole.USER, content="Tell me a joke.")]
        )
        for token in stream_resp:
            text += token.delta
        raw = getattr(stream_resp, "raw", None)
        if raw:
            if getattr(raw, "model", None):
                span.set_attribute("gen_ai.response.model", raw.model)
            if getattr(raw, "id", None):
                span.set_attribute("gen_ai.response.id", raw.id)
        print(f"    -> {text[:60]}")


def run_embeddings_reference():
    """Scenario: embedding generation with reference instrumentation."""
    print("  [embeddings] embedding generation (reference)")
    from llama_index.embeddings.openai import OpenAIEmbedding

    request_model = "text-embedding-3-small"
    embed_model = OpenAIEmbedding(
        model_name=request_model,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )
    with _reference_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        result = embed_model.get_text_embedding("Hello, world!")
        print(f"    -> embedding dim: {len(result)}")


def main():
    print("=== Reference: LlamaIndex Conformance Test ===")

    tp, lp, mp = setup_otel()

    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    request_model = "gpt-4o-mini"
    request_temperature = 0.1
    llm = LlamaOpenAI(
        model=request_model,
        temperature=request_temperature,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )

    run_chat_reference(llm, request_model, request_temperature)
    run_chat_streaming_reference(llm, request_model, request_temperature)
    run_embeddings_reference()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
