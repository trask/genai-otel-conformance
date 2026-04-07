"""Conformance test: prototype instrumentation for AWS Bedrock."""

import json
import os

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

from otel_setup import flush_and_shutdown, setup_otel

from common import run_memory_operations

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
        messages = [
            {
                "role": "user",
                "content": [{"text": "Say hello."}],
            }
        ]
        response = client.converse(
            modelId=request_model,
            messages=messages,
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

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.input.messages": json.dumps([
                {"role": m["role"], "parts": [{"type": "text", "content": m["content"][0]["text"]}]}
                for m in messages
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": text}],
                    "finish_reason": stop_reason,
                }
            ]),
        }
        if stop_reason:
            event_attrs["gen_ai.response.finish_reasons"] = [stop_reason]
        if usage.get("inputTokens") is not None:
            event_attrs["gen_ai.usage.input_tokens"] = usage["inputTokens"]
        if usage.get("outputTokens") is not None:
            event_attrs["gen_ai.usage.output_tokens"] = usage["outputTokens"]
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {text[:60]}")


def run_converse_tool_call_prototype(client):
    """Scenario: Bedrock Converse API with tool calling prototype instrumentation."""
    print("  [chat_tool_call] Bedrock Converse API with tool calling (prototype)")
    request_model = "anthropic.claude-3-haiku-20240307-v1:0"
    tool_spec = {
        "toolSpec": {
            "name": "get_weather",
            "description": "Get the current weather",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                }
            },
        }
    }
    tool_config = {"tools": [tool_spec]}
    with _prototype_tracer.start_as_current_span(
        "chat anthropic.claude-3-haiku-20240307-v1:0"
    ) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps(tool_config["tools"]))
        messages = [
            {
                "role": "user",
                "content": [{"text": "What's the weather in Seattle?"}],
            }
        ]
        response = client.converse(
            modelId=request_model,
            messages=messages,
            toolConfig=tool_config,
        )
        stop_reason = response.get("stopReason")
        if stop_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [stop_reason])
        usage = response.get("usage", {})
        if usage.get("inputTokens") is not None:
            span.set_attribute("gen_ai.usage.input_tokens", usage["inputTokens"])
        if usage.get("outputTokens") is not None:
            span.set_attribute("gen_ai.usage.output_tokens", usage["outputTokens"])
        content = response["output"]["message"]["content"]
        if content and "toolUse" in content[0]:
            print(f"    -> tool_call: {content[0]['toolUse']['name']}")
        else:
            print(f"    -> {content[0]['text'][:60]}")


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
    run_converse_tool_call_prototype(client)
    run_embeddings_prototype(client)
    run_memory_operations(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
