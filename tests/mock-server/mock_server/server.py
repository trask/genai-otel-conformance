"""Mock LLM server for conformance testing.

Provides OpenAI-compatible and Anthropic-compatible endpoints that return
deterministic responses. No real LLM calls are made.
"""

import json
import time
import argparse

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
            resp = dict(OPENAI_CHAT_TOOL_CALL_RESPONSE)
            resp["model"] = body.get("model", resp["model"])
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
