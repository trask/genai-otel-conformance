"""Conformance test: reference instrumentation for AutoGen."""

import asyncio
import contextlib
import os
import time

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_agent_reference():
    """Scenario: basic agent execution via AutoGen with reference instrumentation."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.agents import _base_chat_agent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    print("  [agent_run] basic AutoGen agent execution (reference)")

    request_model = "gpt-4o-mini"
    model_client = OpenAIChatCompletionClient(
        model=request_model,
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )

    previous_create_agent_span = _base_chat_agent.trace_create_agent_span
    previous_invoke_agent_span = _base_chat_agent.trace_invoke_agent_span

    def _disabled_autogen_span(*_args, **_kwargs):
        return contextlib.nullcontext()

    _base_chat_agent.trace_create_agent_span = _disabled_autogen_span
    _base_chat_agent.trace_invoke_agent_span = _disabled_autogen_span

    try:
        with _reference_tracer.start_as_current_span("create_agent test_agent") as span:
            span.set_attribute("gen_ai.operation.name", "create_agent")
            span.set_attribute("gen_ai.provider.name", "openai")
            span.set_attribute("gen_ai.request.model", request_model)
            agent = AssistantAgent(
                name="test_agent",
                model_client=model_client,
                system_message="You are a helpful assistant.",
            )
            span.set_attribute("gen_ai.agent.name", agent.name)
            span.set_attribute("gen_ai.agent.description", agent.description)

        async def _run():
            from autogen_agentchat.messages import TextMessage
            from autogen_core import CancellationToken

            with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
                span.set_attribute("gen_ai.operation.name", "chat")
                span.set_attribute("gen_ai.provider.name", "openai")
                span.set_attribute("gen_ai.request.model", request_model)
                response = await agent.on_messages(
                    [TextMessage(content="Say hello.", source="user")],
                    cancellation_token=CancellationToken(),
                )
                print(f"    -> {str(response.chat_message.content)[:60]}")

        asyncio.run(_run())
    finally:
        _base_chat_agent.trace_create_agent_span = previous_create_agent_span
        _base_chat_agent.trace_invoke_agent_span = previous_invoke_agent_span


def main():
    print("=== Reference: AutoGen Conformance Test ===")

    tp, lp, mp = setup_otel()

    run_agent_reference()

    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
