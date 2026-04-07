"""Shared test infrastructure for AWS Bedrock conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]


def create_bedrock_client():
    """Create a boto3 Bedrock Runtime client pointing at the mock server."""
    import boto3

    client = boto3.client(
        "bedrock-runtime",
        endpoint_url=MOCK_BASE_URL,
        region_name="us-east-1",
        aws_access_key_id="mock",
        aws_secret_access_key="mock",
    )
    return client


def run_converse(client):
    """Scenario: Bedrock Converse API."""
    print("  [converse] Bedrock Converse API")
    response = client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": "Say hello."}],
            }
        ],
    )
    text = response["output"]["message"]["content"][0]["text"]
    print(f"    -> {text[:60]}")


def run_converse_tool_call(client):
    """Scenario: Bedrock Converse API with tool calling."""
    print("  [chat_tool_call] Bedrock Converse API with tool calling")
    tool_config = {
        "tools": [
            {
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
        ],
    }
    response = client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": "What's the weather in Seattle?"}],
            }
        ],
        toolConfig=tool_config,
    )
    content = response["output"]["message"]["content"]
    if content and "toolUse" in content[0]:
        print(f"    -> tool_call: {content[0]['toolUse']['name']}")
    else:
        print(f"    -> {content[0]['text'][:60]}")


def run_embeddings(client):
    """Scenario: Bedrock Titan Embeddings via InvokeModel."""
    import json as _json

    print("  [embeddings] Bedrock Titan Embeddings")
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=_json.dumps({"inputText": "Hello, world!"}),
    )
    result = _json.loads(response["body"].read())
    print(f"    -> embedding dim: {len(result['embedding'])}")


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_bedrock_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
