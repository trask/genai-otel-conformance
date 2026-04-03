"""Conformance test: prototype instrumentation for AWS Bedrock."""

import os

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def create_bedrock_client():
    """Create a boto3 Bedrock Runtime client pointing at the mock server."""
    import boto3

    return boto3.client(
        "bedrock-runtime",
        endpoint_url=MOCK_BASE_URL,
        region_name="us-east-1",
        aws_access_key_id="mock",
        aws_secret_access_key="mock",
    )


def run_converse_prototype(client):
    """Scenario: Bedrock Converse API with prototype instrumentation."""
    print("  [converse] Bedrock Converse API (prototype)")
    request_model = "anthropic.claude-3-haiku-20240307-v1:0"
    with _prototype_tracer.start_as_current_span(
        "chat anthropic.claude-3-haiku-20240307-v1:0"
    ) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")
        span.set_attribute("gen_ai.request.model", request_model)
        response = client.converse(
            modelId=request_model,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Say hello."}],
                }
            ],
        )
        stop_reason = response.get("stopReason")
        if stop_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [stop_reason])
        usage = response.get("usage", {})
        if usage.get("inputTokens") is not None:
            span.set_attribute("gen_ai.usage.input_tokens", usage["inputTokens"])
        if usage.get("outputTokens") is not None:
            span.set_attribute("gen_ai.usage.output_tokens", usage["outputTokens"])
        text = response["output"]["message"]["content"][0]["text"]
        print(f"    -> {text[:60]}")


def run_embeddings_prototype(client):
    """Scenario: Bedrock Titan Embeddings with prototype instrumentation."""
    import json as _json

    print("  [embeddings] Bedrock Titan Embeddings (prototype)")
    request_model = "amazon.titan-embed-text-v2:0"
    with _prototype_tracer.start_as_current_span(
        "embeddings amazon.titan-embed-text-v2:0"
    ) as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")
        span.set_attribute("gen_ai.request.model", request_model)
        response = client.invoke_model(
            modelId=request_model,
            contentType="application/json",
            accept="application/json",
            body=_json.dumps({"inputText": "Hello, world!"}),
        )
        result = _json.loads(response["body"].read())
        if result.get("inputTextTokenCount") is not None:
            span.set_attribute("gen_ai.usage.input_tokens", result["inputTextTokenCount"])
        print(f"    -> embedding dim: {len(result['embedding'])}")


def main():
    print("=== Prototype: AWS Bedrock Conformance Test ===")

    tp, lp, mp = setup_otel()

    client = create_bedrock_client()

    run_converse_prototype(client)
    run_embeddings_prototype(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
