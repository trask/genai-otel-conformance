"""Conformance test: prototype instrumentation for Google ADK."""

import asyncio
import contextlib
import json
import os
import time

from opentelemetry import trace as _trace
from opentelemetry.sdk.trace import SpanProcessor

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = _trace.get_tracer("gen_ai.prototype")


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


@contextlib.contextmanager
def _suppress_adk_native_tracing():
    from google.adk import runners as adk_runners
    from google.adk.agents import base_agent as adk_base_agent
    from google.adk.flows.llm_flows import base_llm_flow as adk_base_llm_flow
    from google.adk.flows.llm_flows import functions as adk_functions
    from google.adk.telemetry import tracing as adk_tracing

    class _DisabledTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, *_args, **_kwargs):
            yield _trace.NonRecordingSpan(_trace.INVALID_SPAN_CONTEXT)

    disabled_tracer = _DisabledTracer()
    patched_modules = (
        adk_tracing,
        adk_base_agent,
        adk_runners,
        adk_base_llm_flow,
        adk_functions,
    )
    previous_tracers = {module: module.tracer for module in patched_modules}
    previous_emit = adk_tracing.otel_logger.emit

    try:
        for module in patched_modules:
            module.tracer = disabled_tracer
        adk_tracing.otel_logger.emit = lambda *_args, **_kwargs: None
        yield
    finally:
        for module, tracer in previous_tracers.items():
            module.tracer = tracer
        adk_tracing.otel_logger.emit = previous_emit


def run_agent_prototype():
    """Scenario: basic agent execution via Google ADK with prototype instrumentation."""
    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    print("  [agent_run] basic ADK agent execution (prototype)")

    os.environ.setdefault("GOOGLE_API_KEY", "mock-key")
    request_model = "gemini-2.0-flash"

    def get_weather(location: str) -> str:
        """Get the current weather."""
        return f"Sunny in {location}"

    tool_defs = [
        {
            "name": "get_weather",
            "description": "Get the current weather.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {"type": "STRING", "description": "City name"},
                },
                "required": ["location"],
            },
        }
    ]

    with _suppress_adk_native_tracing():
        agent = Agent(
            name="test_agent",
            model=Gemini(model=request_model, base_url=MOCK_BASE_URL),
            instruction="You are a helpful assistant.",
            tools=[get_weather],
        )

        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name="test_app", session_service=session_service)

        async def _run():
            session = await session_service.create_session(
                app_name="test_app", user_id="test_user",
            )
            with _prototype_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
                span.set_attribute("gen_ai.provider.name", "google_genai")
                span.set_attribute("gen_ai.conversation.id", session.id)
                span.set_attribute("gen_ai.request.model", request_model)
                span.set_attribute("gen_ai.tool.definitions", json.dumps(tool_defs))
                usage_metadata = None
                finish_reason = None
                try:
                    async for event in runner.run_async(
                        user_id="test_user",
                        session_id=session.id,
                        new_message=types.Content(
                            role="user",
                            parts=[types.Part(text="Say hello.")],
                        ),
                    ):
                        if getattr(event, "usage_metadata", None) is not None:
                            usage_metadata = event.usage_metadata
                        event_finish_reason = getattr(event, "finish_reason", None)
                        if isinstance(event, dict):
                            event_finish_reason = event.get("finish_reason")
                        if event_finish_reason is not None:
                            finish_reason = getattr(event_finish_reason, "value", event_finish_reason)
                        if event.content and event.content.parts:
                            text = event.content.parts[0].text
                            if text:
                                print(f"    -> {text[:60]}")
                    if usage_metadata is not None:
                        prompt_token_count = getattr(usage_metadata, "prompt_token_count", None)
                        candidate_token_count = getattr(usage_metadata, "candidates_token_count", None)
                        thoughts_token_count = getattr(usage_metadata, "thoughts_token_count", None)
                        if isinstance(usage_metadata, dict):
                            prompt_token_count = usage_metadata.get("prompt_token_count")
                            candidate_token_count = usage_metadata.get("candidates_token_count")
                            thoughts_token_count = usage_metadata.get("thoughts_token_count")
                        if prompt_token_count is not None:
                            span.set_attribute("gen_ai.usage.input_tokens", prompt_token_count)
                        if candidate_token_count is not None:
                            span.set_attribute("gen_ai.usage.output_tokens", candidate_token_count)
                        if thoughts_token_count:
                            span.set_attribute("gen_ai.usage.reasoning.output_tokens", thoughts_token_count)
                    if finish_reason is not None:
                        span.set_attribute(
                            "gen_ai.response.finish_reasons",
                            [str(finish_reason).lower()],
                        )
                except Exception as exc:
                    print(f"    [error] agent execution failed: {exc}")

        asyncio.run(_run())


def main():
    print("=== Prototype: Google ADK Conformance Test ===")

    tp, lp, mp = setup_otel()

    span_counter = SpanCounter()
    tp.add_span_processor(span_counter)

    run_agent_prototype()

    print(f"\n  [diagnostic] Spans generated: {span_counter.count}")

    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
