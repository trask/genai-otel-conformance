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
def openai_chat_completions():
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
def openai_embeddings():
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
    yield "["
    for i, chunk in enumerate(chunks):
        if i > 0:
            yield ","
        yield json.dumps(chunk)
    yield "]"


@app.route("/v1beta/models/<path:model_action>", methods=["POST"])
def google_genai(model_action):
    """Handle Google GenAI API requests (generateContent, streamGenerateContent)."""
    if ":streamGenerateContent" in model_action:
        return Response(_stream_google_genai(), mimetype="application/x-ndjson")
    # :generateContent or any other action
    return GOOGLE_GENAI_RESPONSE


@app.route("/v1/projects/<path:rest>", methods=["POST"])
def vertex_ai(rest):
    """Handle Vertex AI API requests (same response format as Google GenAI)."""
    if ":streamGenerateContent" in rest:
        return Response(
            _stream_google_genai_json_array(), mimetype="application/json"
        )
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


def _stream_bedrock_converse():
    """Yield Bedrock ConverseStream event-stream chunks."""
    events = []
    events.append({"messageStart": {"role": "assistant"}})
    for word in ["This ", "is ", "a ", "mock ", "streamed ", "response."]:
        events.append({"contentBlockDelta": {"delta": {"text": word}, "contentBlockIndex": 0}})
    events.append({"contentBlockStop": {"contentBlockIndex": 0}})
    events.append({
        "messageStop": {"stopReason": "end_turn"},
    })
    events.append({
        "metadata": {
            "usage": {"inputTokens": 25, "outputTokens": 6, "totalTokens": 31},
            "metrics": {"latencyMs": 100},
        }
    })
    for event in events:
        yield json.dumps(event) + "\n"


@app.route("/model/<path:model_id>/converse", methods=["POST"])
def bedrock_converse(model_id):
    return BEDROCK_CONVERSE_RESPONSE


@app.route("/model/<path:model_id>/converse-stream", methods=["POST"])
def bedrock_converse_stream(model_id):
    return Response(_stream_bedrock_converse(), mimetype="application/vnd.amazon.eventstream")


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
