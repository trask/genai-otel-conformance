"""Mock LLM server for conformance testing.

Provides OpenAI-compatible and Anthropic-compatible endpoints that return
deterministic responses. No real LLM calls are made.
"""

import argparse
import copy
import json
import time

from flask import Flask, request, Response

app = Flask(__name__)

# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------

OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-mock-001",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock response from the conformance test server.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 12,
        "total_tokens": 37,
    },
}

OPENAI_CHAT_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-mock-002",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mock_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Seattle"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
    },
}

OPENAI_EMBEDDING_RESPONSE = {
    "id": "embd-mock-001",
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.001] * 256,
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {
        "prompt_tokens": 8,
        "total_tokens": 8,
    },
}

OPENAI_RESPONSES_RESPONSE = {
    "id": "resp-mock-001",
    "object": "response",
    "created_at": 1700000000,
    "model": "gpt-4o-mini",
    "output": [
        {
            "type": "message",
            "id": "msg-mock-001",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "This is a mock response from the conformance test server.",
                }
            ],
        }
    ],
    "usage": {
        "input_tokens": 25,
        "output_tokens": 12,
        "total_tokens": 37,
    },
}


def _mock_tool_argument_value(name, schema):
    schema_type = (schema or {}).get("type")
    if name == "location":
        return "Seattle"
    if name == "message":
        return "This is a mock response from the conformance test server."
    if schema_type == "string":
        return f"mock-{name}"
    if schema_type in {"integer", "number"}:
        return 1
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}
    return f"mock-{name}"


def _mock_tool_arguments(tool):
    function = (tool or {}).get("function", {})
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])

    argument_names = list(required) or list(properties)
    if not argument_names:
        return {"value": "mock-value"}

    return {
        name: _mock_tool_argument_value(name, properties.get(name, {}))
        for name in argument_names
    }


def _stream_openai_chat(body):
    """Yield SSE chunks for an OpenAI streaming chat completion."""
    model = body.get("model", "gpt-4o-mini")
    chunk_id = "chatcmpl-mock-stream-001"

    # role chunk
    yield _sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
    )

    # content chunks
    for word in ["This ", "is ", "a ", "mock ", "streamed ", "response."]:
        yield _sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            }
        )

    # usage chunk
    yield _sse(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 6,
                "total_tokens": 31,
            },
        }
    )

    yield "data: [DONE]\n\n"


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/openai/v1/chat/completions", methods=["POST"])
@app.route("/openai/deployments/<deployment>/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def openai_chat_completions(deployment=None):
    body = request.get_json(silent=True) or {}

    # Streaming
    if body.get("stream"):
        return Response(_stream_openai_chat(body), mimetype="text/event-stream")

    # Tool-call detection: if tools are provided and no tool result yet,
    # return a tool call; otherwise return a normal response (completes the
    # agent loop).
    if body.get("tools"):
        messages = body.get("messages", [])
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if not has_tool_result:
            resp = copy.deepcopy(OPENAI_CHAT_TOOL_CALL_RESPONSE)
            resp["model"] = body.get("model", resp["model"])
            tool = body.get("tools", [{}])[0]
            tool_name = tool.get("function", {}).get("name")
            if tool_name:
                resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = tool_name
            resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                _mock_tool_arguments(tool)
            )
            return resp

    resp = dict(OPENAI_CHAT_RESPONSE)
    resp["model"] = body.get("model", resp["model"])
    return resp


@app.route("/v1/embeddings", methods=["POST"])
@app.route("/openai/v1/embeddings", methods=["POST"])
@app.route("/openai/deployments/<deployment>/embeddings", methods=["POST"])
@app.route("/embeddings", methods=["POST"])
def openai_embeddings(deployment=None):
    body = request.get_json(silent=True) or {}
    resp = dict(OPENAI_EMBEDDING_RESPONSE)
    resp["model"] = body.get("model", resp["model"])
    return resp


@app.route("/v1/responses", methods=["POST"])
def openai_responses():
    body = request.get_json(silent=True) or {}
    resp = dict(OPENAI_RESPONSES_RESPONSE)
    resp["model"] = body.get("model", resp["model"])
    return resp


# ---------------------------------------------------------------------------
# Anthropic-compatible endpoints
# ---------------------------------------------------------------------------

ANTHROPIC_MESSAGE_RESPONSE = {
    "id": "msg-mock-001",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "This is a mock response from the conformance test server.",
        }
    ],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 25,
        "output_tokens": 12,
    },
}


def _stream_anthropic_message(body):
    """Yield SSE events for Anthropic streaming."""
    model = body.get("model", "claude-sonnet-4-20250514")

    yield _sse_anthropic("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg-mock-stream-001",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 0},
        },
    })

    yield _sse_anthropic("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    for word in ["This ", "is ", "a ", "mock ", "streamed ", "response."]:
        yield _sse_anthropic("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": word},
        })

    yield _sse_anthropic("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    })

    yield _sse_anthropic("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 6},
    })

    yield _sse_anthropic("message_stop", {
        "type": "message_stop",
    })


def _sse_anthropic(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    body = request.get_json(silent=True) or {}

    if body.get("stream"):
        return Response(_stream_anthropic_message(body), mimetype="text/event-stream")

    resp = dict(ANTHROPIC_MESSAGE_RESPONSE)
    resp["model"] = body.get("model", resp["model"])
    return resp


# ---------------------------------------------------------------------------
# Google GenAI / Vertex AI -compatible endpoints
# ---------------------------------------------------------------------------

GOOGLE_GENAI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": "This is a mock response from the conformance test server."}],
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 25,
        "candidatesTokenCount": 12,
        "totalTokenCount": 37,
    },
    "modelVersion": "gemini-2.0-flash",
}


def _stream_google_genai():
    """Yield line-delimited JSON chunks for Google GenAI streaming."""
    for word in ["This ", "is ", "a ", "mock ", "streamed ", "response."]:
        chunk = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": word}]},
                    "index": 0,
                }
            ],
        }
        yield json.dumps(chunk) + "\n"

    # Final chunk with usage metadata
    final = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": ""}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 6,
            "totalTokenCount": 31,
        },
    }
    yield json.dumps(final) + "\n"


def _stream_google_genai_json_array():
    """Yield a JSON array of chunks for Vertex AI REST streaming.

    The Vertex AI gapic REST transport expects the streaming response body
    to be a JSON array (``[{chunk}, {chunk}, ...]``), not NDJSON.
    """
    chunks = _google_genai_stream_chunks()
    yield "["
    for i, chunk in enumerate(chunks):
        if i > 0:
            yield ","
        yield json.dumps(chunk)
    yield "]"


def _stream_google_genai_sse():
    """Yield SSE-formatted chunks for Vertex AI JS SDK streaming.

    The JS ``@google-cloud/vertexai`` SDK requests ``?alt=sse`` and expects
    ``data: <json>\\n\\n`` framing.
    """
    for chunk in _google_genai_stream_chunks():
        yield f"data: {json.dumps(chunk)}\n\n"


def _google_genai_stream_chunks():
    """Return the list of streaming chunks for Google GenAI / Vertex AI."""
    chunks = []
    for word in ["This ", "is ", "a ", "mock ", "streamed ", "response."]:
        chunks.append({
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": word}]},
                    "index": 0,
                }
            ],
        })
    chunks.append({
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": ""}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 6,
            "totalTokenCount": 31,
        },
    })
    return chunks


GOOGLE_GENAI_EMBEDDING_RESPONSE = {
    "embedding": {
        "values": [0.001] * 256,
    },
}

GOOGLE_GENAI_BATCH_EMBEDDING_RESPONSE = {
    "embeddings": [
        {"values": [0.001] * 256},
    ],
}


@app.route("/v1beta/models/<path:model_action>", methods=["POST"])
def google_genai(model_action):
    """Handle Google GenAI API requests (generateContent, streamGenerateContent, embedContent)."""
    if ":streamGenerateContent" in model_action:
        return Response(_stream_google_genai(), mimetype="application/x-ndjson")
    if ":batchEmbedContents" in model_action:
        return GOOGLE_GENAI_BATCH_EMBEDDING_RESPONSE
    if ":embedContent" in model_action:
        return GOOGLE_GENAI_EMBEDDING_RESPONSE
    # :generateContent or any other action
    return GOOGLE_GENAI_RESPONSE


@app.route("/v1/projects/<path:rest>", methods=["POST"])
def vertex_ai(rest):
    """Handle Vertex AI API requests (same response format as Google GenAI)."""
    if ":streamGenerateContent" in rest:
        if request.args.get("alt") == "sse":
            return Response(
                _stream_google_genai_sse(), mimetype="text/event-stream"
            )
        return Response(
            _stream_google_genai_json_array(), mimetype="application/json"
        )
    if ":predict" in rest:
        # Vertex AI embeddings use the predict endpoint
        body = request.get_json(silent=True) or {}
        instances = body.get("instances", [])
        predictions = []
        for _ in instances:
            predictions.append({"embeddings": {"values": [0.001] * 256}})
        return {
            "predictions": predictions,
            "metadata": {"billableCharacterCount": 13},
        }
    return GOOGLE_GENAI_RESPONSE


# ---------------------------------------------------------------------------
# AWS Bedrock-compatible endpoints
# ---------------------------------------------------------------------------

BEDROCK_CONVERSE_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "This is a mock response from the conformance test server."}
            ],
        }
    },
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 25,
        "outputTokens": 12,
        "totalTokens": 37,
    },
    "metrics": {"latencyMs": 100},
}


def _encode_event_stream_message(event_type, payload_bytes):
    """Encode a single AWS event-stream binary message.

    Format (all big-endian):
      total_length (4) | headers_length (4) | prelude_crc (4)
      headers (variable) | payload (variable) | message_crc (4)
    """
    import struct, binascii

    def _crc32(data):
        return binascii.crc32(data) & 0xFFFFFFFF

    def _encode_header(name, value):
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        # 1 byte name len + name + 1 byte type (7=string) + 2 bytes value len + value
        return (
            struct.pack("!B", len(name_bytes))
            + name_bytes
            + struct.pack("!B", 7)
            + struct.pack("!H", len(value_bytes))
            + value_bytes
        )

    headers = b""
    headers += _encode_header(":message-type", "event")
    headers += _encode_header(":event-type", event_type)
    headers += _encode_header(":content-type", "application/json")

    total_length = 4 + 4 + 4 + len(headers) + len(payload_bytes) + 4
    prelude = struct.pack("!II", total_length, len(headers))
    prelude_crc = struct.pack("!I", _crc32(prelude))
    message_no_crc = prelude + prelude_crc + headers + payload_bytes
    message_crc = struct.pack("!I", _crc32(message_no_crc))
    return message_no_crc + message_crc


def _stream_bedrock_converse():
    """Yield Bedrock ConverseStream event-stream chunks in binary format."""
    events = []
    events.append(("messageStart", {"role": "assistant"}))
    for word in ["This ", "is ", "a ", "mock ", "streamed ", "response."]:
        events.append(("contentBlockDelta", {"delta": {"text": word}, "contentBlockIndex": 0}))
    events.append(("contentBlockStop", {"contentBlockIndex": 0}))
    events.append(("messageStop", {"stopReason": "end_turn"}))
    events.append(("metadata", {
        "usage": {"inputTokens": 25, "outputTokens": 6, "totalTokens": 31},
        "metrics": {"latencyMs": 100},
    }))
    for event_type, body in events:
        payload = json.dumps(body).encode("utf-8")
        yield _encode_event_stream_message(event_type, payload)


@app.route("/model/<path:model_id>/converse", methods=["POST"])
def bedrock_converse(model_id):
    return BEDROCK_CONVERSE_RESPONSE


@app.route("/model/<path:model_id>/converse-stream", methods=["POST"])
def bedrock_converse_stream(model_id):
    return Response(_stream_bedrock_converse(), mimetype="application/vnd.amazon.eventstream")


@app.route("/model/<path:model_id>/invoke", methods=["POST"])
def bedrock_invoke(model_id):
    """Handle Bedrock InvokeModel — used for Titan Embeddings."""
    body = request.get_json(silent=True) or {}
    # Amazon Titan Embeddings response format
    resp = {
        "embedding": [0.001] * 256,
        "inputTextTokenCount": 8,
    }
    return Response(json.dumps(resp), mimetype="application/json")


# ---------------------------------------------------------------------------
# AWS Bedrock Agent Runtime-compatible endpoints
# ---------------------------------------------------------------------------

def _stream_bedrock_agent_invoke():
    """Yield Bedrock Agent invoke_agent event-stream chunks in binary format."""
    import base64
    events = []
    # The agent response is delivered as chunk events with base64-encoded bytes
    text = "This is a mock response from the conformance test server."
    events.append(("chunk", {"bytes": base64.b64encode(text.encode("utf-8")).decode("ascii")}))
    for event_type, body in events:
        payload = json.dumps(body).encode("utf-8")
        yield _encode_event_stream_message(event_type, payload)


@app.route("/agents/<agent_id>/agentAliases/<alias_id>/sessions/<session_id>/text", methods=["POST"])
def bedrock_agent_invoke(agent_id, alias_id, session_id):
    """Handle Bedrock Agent Runtime InvokeAgent."""
    return Response(
        _stream_bedrock_agent_invoke(),
        mimetype="application/vnd.amazon.eventstream",
        headers={
            "x-amzn-bedrock-agent-session-id": session_id,
            "x-amz-bedrock-agent-content-type": "application/json",
        },
    )


# ---------------------------------------------------------------------------
# AWS Bedrock AgentCore Memory endpoints
# ---------------------------------------------------------------------------

# In-memory store for Bedrock AgentCore memory operations.
_AGENTCORE_MEMORY_RECORDS: dict[str, list[dict]] = {}  # memoryId -> records
_AGENTCORE_RECORD_COUNTER = 0


def _make_memory_record_id():
    """Generate a mock memory record ID (min 40 chars per SDK validation)."""
    global _AGENTCORE_RECORD_COUNTER
    _AGENTCORE_RECORD_COUNTER += 1
    return f"mr-mock-{_AGENTCORE_RECORD_COUNTER:03d}-{'0' * 28}"


@app.route("/memories/<memory_id>/memoryRecords/batchCreate", methods=["POST"])
def bedrock_batch_create_memory_records(memory_id):
    """Mock Bedrock AgentCore BatchCreateMemoryRecords."""
    body = request.get_json(silent=True) or {}
    records = body.get("records", [])
    if memory_id not in _AGENTCORE_MEMORY_RECORDS:
        _AGENTCORE_MEMORY_RECORDS[memory_id] = []
    successful = []
    for rec in records:
        record_id = _make_memory_record_id()
        stored = {
            "memoryRecordId": record_id,
            "content": rec.get("content", {}),
        }
        _AGENTCORE_MEMORY_RECORDS[memory_id].append(stored)
        successful.append({
            "memoryRecordId": record_id,
            "status": "COMPLETED",
            "requestIdentifier": rec.get("requestIdentifier", ""),
        })
    return Response(
        json.dumps({"successfulRecords": successful, "failedRecords": []}),
        status=201, mimetype="application/json",
    )


@app.route("/memories/<memory_id>/retrieve", methods=["POST"])
def bedrock_retrieve_memory_records(memory_id):
    """Mock Bedrock AgentCore RetrieveMemoryRecords."""
    recs = _AGENTCORE_MEMORY_RECORDS.get(memory_id, [])
    summaries = []
    for rec in recs[-5:]:
        summaries.append({
            "memoryRecordId": rec["memoryRecordId"],
            "content": rec.get("content", {}),
            "score": 0.95,
        })
    return {"memoryRecordSummaries": summaries}


@app.route("/memories/<memory_id>/memoryRecords/batchDelete", methods=["POST"])
def bedrock_batch_delete_memory_records(memory_id):
    """Mock Bedrock AgentCore BatchDeleteMemoryRecords."""
    body = request.get_json(silent=True) or {}
    to_delete = {r["memoryRecordId"] for r in body.get("records", [])}
    recs = _AGENTCORE_MEMORY_RECORDS.get(memory_id, [])
    removed = [r for r in recs if r["memoryRecordId"] in to_delete]
    _AGENTCORE_MEMORY_RECORDS[memory_id] = [r for r in recs if r["memoryRecordId"] not in to_delete]
    return {
        "successfulRecords": [{"memoryRecordId": r["memoryRecordId"], "status": "COMPLETED"} for r in removed],
        "failedRecords": [],
    }


# Bedrock AgentCore Control Plane (CreateMemory / DeleteMemory)
_AGENTCORE_MEMORIES: dict[str, dict] = {}  # memoryId -> memory details


@app.route("/memories/create", methods=["POST"])
def bedrock_create_memory():
    """Mock Bedrock AgentCore Control Plane CreateMemory."""
    body = request.get_json(silent=True) or {}
    name = body.get("name", "unnamed-memory")
    memory_id = f"mem-mock-{len(_AGENTCORE_MEMORIES) + 1:03d}"
    memory = {
        "id": memory_id,
        "arn": f"arn:aws:bedrock:us-east-1:123456789012:memory/{memory_id}",
        "name": name,
        "description": body.get("description", ""),
        "status": "ACTIVE",
        "createdAt": 1735689600.0,
        "updatedAt": 1735689600.0,
        "strategies": [],
    }
    _AGENTCORE_MEMORIES[memory_id] = memory
    return Response(
        json.dumps({"memory": memory}),
        status=201, mimetype="application/json",
    )


@app.route("/memories/<memory_id>/delete", methods=["DELETE"])
def bedrock_delete_memory(memory_id):
    """Mock Bedrock AgentCore Control Plane DeleteMemory."""
    _AGENTCORE_MEMORIES.pop(memory_id, None)
    return {"memoryId": memory_id, "status": "DELETING"}
# ---------------------------------------------------------------------------
# Cohere-compatible endpoints
# ---------------------------------------------------------------------------

@app.route("/v2/chat", methods=["POST"])
def cohere_chat():
    body = request.get_json(silent=True) or {}
    return {
        "id": "cohere-mock-001",
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This is a mock response from the conformance test server."}
            ],
        },
        "usage": {
            "billed_units": {"input_tokens": 25, "output_tokens": 12},
            "tokens": {"input_tokens": 25, "output_tokens": 12},
        },
    }


@app.route("/v1/chat", methods=["POST"])
def cohere_chat_v1():
    body = request.get_json(silent=True) or {}
    return {
        "text": "This is a mock response from the conformance test server.",
        "generation_id": "cohere-mock-001",
        "finish_reason": "COMPLETE",
        "meta": {
            "tokens": {"input_tokens": 25, "output_tokens": 12},
            "billed_units": {"input_tokens": 25, "output_tokens": 12},
        },
    }


@app.route("/v2/embed", methods=["POST"])
def cohere_embed():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts", ["Hello, world!"])
    embeddings = [[0.001] * 256 for _ in texts]
    return {
        "id": "cohere-embed-mock-001",
        "embeddings": {"float": embeddings},
        "texts": texts,
        "meta": {
            "api_version": {"version": "2"},
            "billed_units": {"input_tokens": 8},
        },
    }


@app.route("/v1/embed", methods=["POST"])
def cohere_embed_v1():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts", ["Hello, world!"])
    embeddings = [[0.001] * 256 for _ in texts]
    return {
        "response_type": "embeddings_floats",
        "id": "cohere-embed-mock-001",
        "embeddings": embeddings,
        "texts": texts,
        "meta": {
            "api_version": {"version": "1"},
            "billed_units": {"input_tokens": 8},
        },
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# OpenAI Assistants / Azure AI Foundry Agents -compatible endpoints
# ---------------------------------------------------------------------------

# Shared state to track which run has been polled (for completing on second poll)
_run_poll_count: dict[str, int] = {}


@app.route("/v1/assistants", methods=["POST"])
@app.route("/openai/assistants", methods=["POST"])
@app.route("/assistants", methods=["POST"])
def create_assistant():
    body = request.get_json(silent=True) or {}
    return {
        "id": "asst-mock-001",
        "object": "assistant",
        "created_at": 1700000000,
        "name": body.get("name", "mock-assistant"),
        "description": body.get("description"),
        "model": body.get("model", "gpt-4o-mini"),
        "instructions": body.get("instructions", ""),
        "tools": body.get("tools", []),
        "metadata": body.get("metadata", {}),
    }


@app.route("/v1/assistants/<assistant_id>", methods=["DELETE"])
@app.route("/openai/assistants/<assistant_id>", methods=["DELETE"])
@app.route("/assistants/<assistant_id>", methods=["DELETE"])
def delete_assistant(assistant_id):
    return {
        "id": assistant_id,
        "object": "assistant.deleted",
        "deleted": True,
    }


@app.route("/v1/threads", methods=["POST"])
@app.route("/openai/threads", methods=["POST"])
@app.route("/threads", methods=["POST"])
def create_thread():
    return {
        "id": "thread-mock-001",
        "object": "thread",
        "created_at": 1700000000,
        "metadata": {},
    }


@app.route("/v1/threads/<thread_id>/messages", methods=["POST"])
@app.route("/openai/threads/<thread_id>/messages", methods=["POST"])
@app.route("/threads/<thread_id>/messages", methods=["POST"])
def create_message(thread_id):
    body = request.get_json(silent=True) or {}
    return {
        "id": "msg-mock-001",
        "object": "thread.message",
        "created_at": 1700000000,
        "thread_id": thread_id,
        "role": body.get("role", "user"),
        "content": [
            {
                "type": "text",
                "text": {"value": body.get("content", ""), "annotations": []},
            }
        ],
        "metadata": {},
    }


@app.route("/v1/threads/<thread_id>/runs", methods=["POST"])
@app.route("/openai/threads/<thread_id>/runs", methods=["POST"])
@app.route("/threads/<thread_id>/runs", methods=["POST"])
def create_run(thread_id):
    body = request.get_json(silent=True) or {}
    run_id = "run-mock-001"
    return {
        "id": run_id,
        "object": "thread.run",
        "created_at": 1700000000,
        "thread_id": thread_id,
        "assistant_id": body.get("assistant_id", "asst-mock-001"),
        "status": "completed",
        "model": body.get("model", "gpt-4o-mini"),
        "instructions": body.get("instructions"),
        "tools": body.get("tools", []),
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 12,
            "total_tokens": 37,
        },
        "metadata": {},
    }


@app.route("/v1/threads/<thread_id>/runs/<run_id>", methods=["GET"])
@app.route("/openai/threads/<thread_id>/runs/<run_id>", methods=["GET"])
@app.route("/threads/<thread_id>/runs/<run_id>", methods=["GET"])
def get_run(thread_id, run_id):
    return {
        "id": run_id,
        "object": "thread.run",
        "created_at": 1700000000,
        "thread_id": thread_id,
        "assistant_id": "asst-mock-001",
        "status": "completed",
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 12,
            "total_tokens": 37,
        },
        "metadata": {},
    }


@app.route("/v1/threads/<thread_id>/messages", methods=["GET"])
@app.route("/openai/threads/<thread_id>/messages", methods=["GET"])
@app.route("/threads/<thread_id>/messages", methods=["GET"])
def list_messages(thread_id):
    return {
        "object": "list",
        "data": [
            {
                "id": "msg-mock-002",
                "object": "thread.message",
                "created_at": 1700000001,
                "thread_id": thread_id,
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": {
                            "value": "This is a mock response from the conformance test server.",
                            "annotations": [],
                        },
                    }
                ],
                "metadata": {},
            }
        ],
        "first_id": "msg-mock-002",
        "last_id": "msg-mock-002",
        "has_more": False,
    }


# ---------------------------------------------------------------------------
# Mem0-compatible memory endpoints
# ---------------------------------------------------------------------------

# In-memory store for mock memory records.
_MEMORY_RECORDS: list[dict] = []
_MEMORY_COUNTER = 0


@app.route("/v1/ping/", methods=["GET"])
def mem0_ping():
    """Mock Mem0 API key validation endpoint."""
    return {"status": "ok"}, 200


@app.route("/v1/memories/", methods=["POST"])
def mem0_add_memory():
    """Mock Mem0 add memory (POST /v1/memories/)."""
    global _MEMORY_COUNTER
    body = request.get_json(silent=True) or {}
    _MEMORY_COUNTER += 1
    record_id = f"mem_mock_{_MEMORY_COUNTER:03d}"
    record = {
        "id": record_id,
        "memory": body.get("messages", [{}])[0].get("content", "") if body.get("messages") else "",
        "user_id": body.get("user_id"),
        "agent_id": body.get("agent_id"),
        "run_id": body.get("run_id"),
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    _MEMORY_RECORDS.append(record)
    return {"results": [{"id": record_id, "event": "ADD", "memory": record["memory"]}]}


@app.route("/v1/memories/search/", methods=["POST"])
def mem0_search_memory():
    """Mock Mem0 search memory (POST /v1/memories/search/)."""
    body = request.get_json(silent=True) or {}
    results = []
    for rec in _MEMORY_RECORDS[-5:]:
        results.append({
            "id": rec["id"],
            "memory": rec["memory"],
            "score": 0.95,
            "user_id": rec.get("user_id"),
            "agent_id": rec.get("agent_id"),
            "created_at": rec["created_at"],
            "updated_at": rec["updated_at"],
        })
    return {"results": results}


@app.route("/v1/memories/<memory_id>/", methods=["DELETE"])
def mem0_delete_memory(memory_id):
    """Mock Mem0 delete memory (DELETE /v1/memories/<id>/)."""
    global _MEMORY_RECORDS
    _MEMORY_RECORDS = [r for r in _MEMORY_RECORDS if r["id"] != memory_id]
    return {"message": "Memory deleted successfully"}


@app.route("/v1/memories/", methods=["GET"])
def mem0_list_memories():
    """Mock Mem0 list memories (GET /v1/memories/)."""
    results = []
    for rec in _MEMORY_RECORDS:
        results.append({
            "id": rec["id"],
            "memory": rec["memory"],
            "user_id": rec.get("user_id"),
            "agent_id": rec.get("agent_id"),
            "created_at": rec["created_at"],
            "updated_at": rec["updated_at"],
        })
    return {"results": results}


@app.route("/v1/memories/", methods=["DELETE"])
def mem0_delete_all_memories():
    """Mock Mem0 delete all memories (DELETE /v1/memories/)."""
    global _MEMORY_RECORDS
    _MEMORY_RECORDS.clear()
    return {"message": "Memories deleted successfully"}


# ---------------------------------------------------------------------------
# Azure AI Foundry Memory Store endpoints
# ---------------------------------------------------------------------------

_AZURE_MEMORY_STORES: dict[str, dict] = {}  # name -> store details
_AZURE_MEMORY_ITEMS: dict[str, list[dict]] = {}  # "name/scope" -> items
_AZURE_UPDATE_COUNTER = 0


@app.route("/memory_stores", methods=["POST"])
def azure_create_memory_store():
    """Mock Azure AI Foundry create memory store."""
    body = request.get_json(silent=True) or {}
    name = body.get("name", "unnamed-store")
    store = {
        "object": "memory_store",
        "id": f"ms_{name}",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "name": name,
        "description": body.get("description", ""),
        "metadata": body.get("metadata", {}),
        "definition": body.get("definition", {"kind": "default"}),
    }
    _AZURE_MEMORY_STORES[name] = store
    return Response(json.dumps(store), status=200, mimetype="application/json")


@app.route("/memory_stores/<name>:update_memories", methods=["POST"])
def azure_update_memories(name):
    """Mock Azure AI Foundry update memories (LRO-style, returns completed immediately)."""
    global _AZURE_UPDATE_COUNTER
    _AZURE_UPDATE_COUNTER += 1
    update_id = f"upd-mock-{_AZURE_UPDATE_COUNTER:03d}"

    body = request.get_json(silent=True) or {}
    scope = body.get("scope", "default")
    items = body.get("items", [])
    store_scope_key = f"{name}/{scope}"

    if store_scope_key not in _AZURE_MEMORY_ITEMS:
        _AZURE_MEMORY_ITEMS[store_scope_key] = []

    # Create memory operations from conversation items
    operations = []
    for item in items if isinstance(items, list) else []:
        content = item.get("content", "") if isinstance(item, dict) else str(item)
        memory_id = f"mem-{name}-{len(_AZURE_MEMORY_ITEMS[store_scope_key]) + 1:03d}"
        memory_item = {
            "memory_id": memory_id,
            "updated_at": "2025-01-01T00:00:00Z",
            "scope": scope,
            "content": content,
            "kind": "user_profile",
        }
        _AZURE_MEMORY_ITEMS[store_scope_key].append(memory_item)
        operations.append({
            "kind": "create",
            "memory_item": memory_item,
        })

    return Response(
        json.dumps({
            "update_id": update_id,
            "status": "completed",
            "result": {
                "memory_operations": operations,
                "usage": {
                    "embedding_tokens": 10,
                    "input_tokens": 25,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 40,
                },
            },
        }),
        status=202,
        mimetype="application/json",
    )


@app.route("/memory_stores/<name>/updates/<update_id>", methods=["GET"])
def azure_get_update_status(name, update_id):
    """Mock Azure AI Foundry LRO poll for update_memories."""
    return {
        "update_id": update_id,
        "status": "completed",
        "result": {
            "memory_operations": [],
            "usage": {
                "embedding_tokens": 0,
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 0,
            },
        },
    }


@app.route("/memory_stores/<name>:search_memories", methods=["POST"])
def azure_search_memories(name):
    """Mock Azure AI Foundry search memories."""
    body = request.get_json(silent=True) or {}
    scope = body.get("scope", "default")
    store_scope_key = f"{name}/{scope}"

    memories = []
    for item in _AZURE_MEMORY_ITEMS.get(store_scope_key, [])[-5:]:
        memories.append({"memory_item": item})

    return {
        "search_id": "search-mock-001",
        "memories": memories,
        "usage": {
            "embedding_tokens": 5,
            "input_tokens": 10,
            "output_tokens": 0,
            "total_tokens": 15,
        },
    }


@app.route("/memory_stores/<name>:delete_scope", methods=["POST"])
def azure_delete_scope(name):
    """Mock Azure AI Foundry delete scope."""
    body = request.get_json(silent=True) or {}
    scope = body.get("scope", "default")
    store_scope_key = f"{name}/{scope}"
    _AZURE_MEMORY_ITEMS.pop(store_scope_key, None)
    return {
        "object": "memory_store.scope.deleted",
        "name": name,
        "scope": scope,
        "deleted": True,
    }


@app.route("/memory_stores/<name>", methods=["DELETE"])
def azure_delete_memory_store(name):
    """Mock Azure AI Foundry delete memory store."""
    _AZURE_MEMORY_STORES.pop(name, None)
    # Also clean up any scope data
    keys_to_remove = [k for k in _AZURE_MEMORY_ITEMS if k.startswith(f"{name}/")]
    for k in keys_to_remove:
        del _AZURE_MEMORY_ITEMS[k]
    return {
        "object": "memory_store.deleted",
        "name": name,
        "deleted": True,
    }


# ---------------------------------------------------------------------------
# Letta (MemGPT) endpoints
# ---------------------------------------------------------------------------

_LETTA_AGENTS: dict[str, dict] = {}  # agent_id -> agent state
_LETTA_PASSAGES: dict[str, list[dict]] = {}  # agent_id -> passages
_LETTA_PASSAGE_COUNTER = 0
_LETTA_BLOCK_COUNTER = 0


@app.route("/v1/health/", methods=["GET"])
def letta_health():
    """Mock Letta health endpoint."""
    return {"status": "ok"}


@app.route("/v1/agents/", methods=["POST"])
def letta_create_agent():
    """Mock Letta create agent (POST /v1/agents/)."""
    global _LETTA_BLOCK_COUNTER
    body = request.get_json(silent=True) or {}
    agent_id = f"agent-mock-{len(_LETTA_AGENTS) + 1:03d}"

    blocks = []
    for mb in body.get("memory_blocks", []):
        _LETTA_BLOCK_COUNTER += 1
        block_id = f"block-mock-{_LETTA_BLOCK_COUNTER:03d}"
        blocks.append({
            "id": block_id,
            "label": mb.get("label", "default"),
            "value": mb.get("value", ""),
            "limit": 5000,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        })

    agent_state = {
        "id": agent_id,
        "name": body.get("name", f"test-agent-{len(_LETTA_AGENTS) + 1}"),
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "memory": {"blocks": blocks},
        "agent_type": "memgpt_agent",
        "model": body.get("model", "openai/gpt-4o-mini"),
        "embedding": body.get("embedding", "openai/text-embedding-3-small"),
        "tools": [],
        "tags": [],
    }
    _LETTA_AGENTS[agent_id] = agent_state
    _LETTA_PASSAGES[agent_id] = []
    return Response(json.dumps(agent_state), status=200, mimetype="application/json")


@app.route("/v1/agents/<agent_id>", methods=["DELETE"])
def letta_delete_agent(agent_id):
    """Mock Letta delete agent."""
    _LETTA_AGENTS.pop(agent_id, None)
    _LETTA_PASSAGES.pop(agent_id, None)
    return Response(json.dumps({"id": agent_id, "deleted": True}),
                    status=200, mimetype="application/json")


@app.route("/v1/agents/<agent_id>/core-memory/blocks/<block_label>", methods=["PATCH"])
def letta_update_block(agent_id, block_label):
    """Mock Letta update core memory block."""
    body = request.get_json(silent=True) or {}
    agent = _LETTA_AGENTS.get(agent_id, {})
    blocks = agent.get("memory", {}).get("blocks", [])

    for block in blocks:
        if block["label"] == block_label:
            if "value" in body:
                block["value"] = body["value"]
            block["updated_at"] = "2025-01-01T00:00:01Z"
            return Response(json.dumps(block), status=200, mimetype="application/json")

    return Response(json.dumps({"error": f"Block '{block_label}' not found"}),
                    status=404, mimetype="application/json")


@app.route("/v1/agents/<agent_id>/archival-memory", methods=["POST"])
def letta_create_passage(agent_id):
    """Mock Letta create archival memory passage."""
    global _LETTA_PASSAGE_COUNTER
    body = request.get_json(silent=True) or {}
    _LETTA_PASSAGE_COUNTER += 1
    passage_id = f"passage-mock-{_LETTA_PASSAGE_COUNTER:03d}"

    passage = {
        "id": passage_id,
        "text": body.get("text", ""),
        "agent_id": agent_id,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    if agent_id not in _LETTA_PASSAGES:
        _LETTA_PASSAGES[agent_id] = []
    _LETTA_PASSAGES[agent_id].append(passage)
    return Response(json.dumps(passage), status=200, mimetype="application/json")


@app.route("/v1/agents/<agent_id>/archival-memory/search", methods=["GET"])
def letta_search_passages(agent_id):
    """Mock Letta search archival memory."""
    passages = _LETTA_PASSAGES.get(agent_id, [])
    results = [{"id": p["id"], "text": p["text"], "score": 0.95,
                "agent_id": agent_id, "created_at": p["created_at"]}
               for p in passages[-5:]]
    return Response(json.dumps(results), status=200, mimetype="application/json")


@app.route("/v1/agents/<agent_id>/archival-memory/<memory_id>", methods=["DELETE"])
def letta_delete_passage(agent_id, memory_id):
    """Mock Letta delete archival memory passage."""
    if agent_id in _LETTA_PASSAGES:
        _LETTA_PASSAGES[agent_id] = [
            p for p in _LETTA_PASSAGES[agent_id] if p["id"] != memory_id
        ]
    return Response(json.dumps({"id": memory_id, "deleted": True}),
                    status=200, mimetype="application/json")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Mock LLM server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
