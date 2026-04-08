"""Conformance test: prototype instrumentation for AutoGen."""

import asyncio
import contextlib
import json
import os
import time
from urllib.parse import urlparse

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_agent_prototype():
    """Scenario: basic agent execution via AutoGen with prototype instrumentation."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.agents import _base_chat_agent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    print("  [agent_run] basic AutoGen agent execution (prototype)")

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
        with _prototype_tracer.start_as_current_span("create_agent test_agent") as span:
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

            with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
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


def run_chat_tool_call_prototype():
    """Scenario: chat with tool calling with prototype instrumentation."""
    import openai

    print("  [chat_tool_call] chat with tool calling (prototype)")
    request_model = "gpt-4o-mini"
    request_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }
    tools = [request_tool]

    client = openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([
            {
                "type": t["type"],
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in tools
        ]))
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        resp = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
            tools=tools,
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in resp.choices])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
        choice = resp.choices[0]
        if choice.message.tool_calls:
            print(f"    -> tool_call: {choice.message.tool_calls[0].function.name}")
        else:
            print(f"    -> {choice.message.content[:60]}")


def main():
    print("=== Prototype: AutoGen Conformance Test ===")

    tp, lp, mp = setup_otel()

    run_agent_prototype()
    run_chat_tool_call_prototype()

    time.sleep(2)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
