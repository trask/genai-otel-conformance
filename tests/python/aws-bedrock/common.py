"""Shared test infrastructure for AWS Bedrock conformance tests."""

import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]
_parsed = urlparse(MOCK_BASE_URL)
_SERVER_ADDRESS = _parsed.hostname or "localhost"
_SERVER_PORT = _parsed.port or 443

tracer = trace.get_tracer("gen_ai.memory.aws_bedrock")


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


def create_agentcore_client():
    """Create a boto3 Bedrock AgentCore client pointing at the mock server."""
    import boto3

    client = boto3.client(
        "bedrock-agentcore",
        endpoint_url=MOCK_BASE_URL,
        region_name="us-east-1",
        aws_access_key_id="mock",
        aws_secret_access_key="mock",
    )
    return client


def create_agentcore_control_client():
    """Create a boto3 Bedrock AgentCore Control client pointing at the mock server."""
    import boto3

    client = boto3.client(
        "bedrock-agentcore-control",
        endpoint_url=MOCK_BASE_URL,
        region_name="us-east-1",
        aws_access_key_id="mock",
        aws_secret_access_key="mock",
    )
    return client


def run_memory_operations(client):
    """Exercise Bedrock AgentCore Memory operations for conformance testing.

    Uses the actual boto3 bedrock-agentcore SDK with prototype OTel spans
    to demonstrate which gen_ai.memory.* attributes are capturable.
    """
    import datetime

    agentcore = create_agentcore_client()
    control = create_agentcore_control_client()
    memory_name = "conformance-test-memory-store"
    event_expiry_duration = 86400

    def _set_common_attrs(span, operation_name, memory_id):
        span.set_attribute("gen_ai.operation.name", operation_name)
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")
        span.set_attribute("gen_ai.memory.store.id", memory_id)
        span.set_attribute("server.address", _SERVER_ADDRESS)
        span.set_attribute("server.port", _SERVER_PORT)

    # 0. Create memory store (create_memory_store span)
    print("  [create_memory_store] Bedrock AgentCore CreateMemory")
    with tracer.start_as_current_span("create_memory_store", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "create_memory_store")
        span.set_attribute("gen_ai.provider.name", "aws.bedrock")
        span.set_attribute("server.address", _SERVER_ADDRESS)
        span.set_attribute("server.port", _SERVER_PORT)
        expiration = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=event_expiry_duration)
        span.set_attribute("gen_ai.memory.expiration_date", expiration.isoformat())
        try:
            create_memory_resp = control.create_memory(
                name=memory_name,
                eventExpiryDuration=event_expiry_duration,
            )
            memory = create_memory_resp["memory"]
            memory_id = memory["id"]
            span.set_attribute("gen_ai.memory.store.id", memory_id)
            print(f"    -> created memory store: {memory_id} ({memory['name']})")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 1. Create memory records (update_memory span)
    print("  [update_memory] Bedrock AgentCore BatchCreateMemoryRecords")
    now = datetime.datetime.now(datetime.timezone.utc)
    records_input = [
        {
            "requestIdentifier": "req-001",
            "namespaces": ["conformance-test"],
            "content": {"text": "The user prefers concise answers."},
            "timestamp": now,
        },
        {
            "requestIdentifier": "req-002",
            "namespaces": ["conformance-test"],
            "content": {"text": "The user's name is Alice."},
            "timestamp": now,
        },
    ]
    with tracer.start_as_current_span("update_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "update_memory", memory_id)
        content_parts = [r["content"]["text"] for r in records_input]
        span.set_attribute("gen_ai.memory.record.content", "; ".join(content_parts))
        try:
            create_resp = agentcore.batch_create_memory_records(
                memoryId=memory_id,
                records=records_input,
            )
            record_ids = [r["memoryRecordId"] for r in create_resp["successfulRecords"]]
            if record_ids:
                span.set_attribute("gen_ai.memory.record.id", record_ids[0])
            print(f"    -> created {len(record_ids)} records: {record_ids}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 2. Retrieve memory records (search_memory span)
    print("  [search_memory] Bedrock AgentCore RetrieveMemoryRecords")
    search_query = "What does the user prefer?"
    with tracer.start_as_current_span("search_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "search_memory", memory_id)
        span.set_attribute("gen_ai.memory.query.text", search_query)
        try:
            retrieve_resp = agentcore.retrieve_memory_records(
                memoryId=memory_id,
                namespace="conformance-test",
                searchCriteria={"searchQuery": search_query},
            )
            summaries = retrieve_resp["memoryRecordSummaries"]
            span.set_attribute("gen_ai.memory.search.result.count", len(summaries))
            print(f"    -> retrieved {len(summaries)} records")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 3. Delete memory records (delete_memory span)
    print("  [delete_memory] Bedrock AgentCore BatchDeleteMemoryRecords")
    with tracer.start_as_current_span("delete_memory", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "delete_memory", memory_id)
        if record_ids:
            span.set_attribute("gen_ai.memory.record.id", record_ids[0])
        try:
            delete_resp = agentcore.batch_delete_memory_records(
                memoryId=memory_id,
                records=[{"memoryRecordId": rid} for rid in record_ids[:1]],
            )
            print(f"    -> deleted {len(delete_resp['successfulRecords'])} records")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    # 4. Delete memory store (delete_memory_store span)
    print("  [delete_memory_store] Bedrock AgentCore DeleteMemory")
    with tracer.start_as_current_span("delete_memory_store", kind=SpanKind.CLIENT) as span:
        _set_common_attrs(span, "delete_memory_store", memory_id)
        try:
            control.delete_memory(memoryId=memory_id)
            print(f"    -> deleted memory store: {memory_id}")
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.set_attribute("error.type", type(exc).__qualname__)
            raise

    agentcore.close()
    control.close()


def run(title, instrument_fn, scenarios):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    client = create_bedrock_client()

    for scenario in scenarios:
        scenario(client)

    flush_and_shutdown(tp, lp, mp)
