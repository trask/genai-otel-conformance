"""Conformance test: OTel contrib opentelemetry-instrumentation-claude-agent-sdk.

Exercises: basic agent query via the Claude Agent SDK with a mock transport,
instrumented by the official OTel Claude Agent SDK instrumentor.

NOTE: The Claude Agent SDK communicates via a subprocess transport to the
Claude Code CLI, not via HTTP. A MockTransport is used so the test can run
without the CLI or a live Anthropic API key.

KNOWN LIMITATION — EMPTY RESULTS EXPECTED:
    The OTel instrumentation package (opentelemetry-instrumentation-claude-agent-sdk)
    is not yet published on PyPI and the instrumentor is currently a stub — it
    registers tracer/logger/meter providers but does NOT patch any SDK methods.
    Consequently it produces 0 spans.  This is NOT a test-infrastructure bug;
    the instrumentor simply has no hooks implemented yet.  This test is
    structured to produce telemetry once the instrumentation matures and actual
    monkey-patching of claude_agent_sdk entry-points is added.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from opentelemetry.sdk.trace import SpanProcessor

from otel_setup import setup_otel, flush_and_shutdown


class SpanCounter(SpanProcessor):
    """Lightweight span counter for diagnosing whether instrumentation fires."""

    def __init__(self):
        self.count = 0

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        self.count += 1

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


def instrument():
    """Instrument the Claude Agent SDK. Skips gracefully if not installed."""
    try:
        from opentelemetry.instrumentation.claude_agent_sdk import (
            ClaudeAgentSDKInstrumentor,
        )

        ClaudeAgentSDKInstrumentor().instrument()
        print("  [instrument] ClaudeAgentSDKInstrumentor activated")
    except ImportError:
        print(
            "  [instrument] SKIP: opentelemetry-instrumentation-claude-agent-sdk "
            "not installed (not yet on PyPI — install from source)"
        )


class MockTransport:
    """A mock Transport that simulates the Claude Code CLI protocol.

    This allows the SDK's query() to run end-to-end without spawning
    a real CLI subprocess or making any network calls.
    """

    def __init__(self):
        self._ready = False
        self._pending_responses: list[dict[str, Any]] = []
        self._input_ended = False

    async def connect(self) -> None:
        self._ready = True

    async def write(self, data: str) -> None:
        msg = json.loads(data.strip())

        if msg.get("type") == "control_request":
            request_id = msg["request_id"]
            subtype = msg.get("request", {}).get("subtype", "")
            self._pending_responses.append(
                {
                    "type": "control_response",
                    "response": {
                        "request_id": request_id,
                        "subtype": subtype,
                        "response": {},
                    },
                }
            )

        elif msg.get("type") == "user":
            self._pending_responses.append(
                {
                    "type": "assistant",
                    "message": {
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hello! I'm a mock Claude response.",
                            }
                        ],
                    },
                }
            )
            self._pending_responses.append(
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 100,
                    "duration_api_ms": 50,
                    "is_error": False,
                    "num_turns": 1,
                    "session_id": "mock-session-001",
                    "stop_reason": "end_turn",
                    "total_cost_usd": 0.001,
                    "usage": {"input_tokens": 10, "output_tokens": 8},
                }
            )

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            while not self._pending_responses:
                if self._input_ended:
                    return
                await asyncio.sleep(0.01)
            yield self._pending_responses.pop(0)

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        self._input_ended = True

    async def close(self) -> None:
        self._ready = False


async def run_agent_query():
    """Scenario: basic agent query with mock transport."""
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        query,
    )

    print("  [agent_query] basic query via mock transport")
    transport = MockTransport()
    async for message in query(prompt="Say hello.", transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"    -> {block.text[:60]}")
        elif isinstance(message, ResultMessage):
            print(
                f"    -> result: turns={message.num_turns}, "
                f"model_session={message.session_id}"
            )


def main():
    print("=== OTel Contrib: Claude Agent SDK Conformance Test ===")

    tp, lp, mp = setup_otel()

    span_counter = SpanCounter()
    tp.add_span_processor(span_counter)

    instrument()

    asyncio.run(run_agent_query())

    print(f"\n  [diagnostic] Spans generated: {span_counter.count}")
    if span_counter.count == 0:
        print(
            "  [diagnostic] 0 spans — expected: the "
            "opentelemetry-instrumentation-claude-agent-sdk package is "
            "currently a stub and does not patch any SDK methods yet."
        )

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
