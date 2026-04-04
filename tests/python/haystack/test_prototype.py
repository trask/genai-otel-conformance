"""Conformance test: prototype instrumentation for Haystack.

Exercises: chat via Haystack pipeline
against a mock OpenAI server, with manual OTel spans.
"""

import json
import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat via Haystack pipeline with prototype instrumentation."""
    print("  [chat] basic chat via Haystack pipeline (prototype)")
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    os.environ["OPENAI_API_KEY"] = "mock-key"
    request_model = "gpt-4o-mini"

    generator = OpenAIChatGenerator(
        model=request_model,
        api_base_url=MOCK_BASE_URL,
    )

    messages = [ChatMessage.from_user("Say hello.")]

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        result = generator.run(messages=messages)
        reply = result["replies"][0]
        if hasattr(reply, "meta") and reply.meta:
            meta = reply.meta
            if "model" in meta:
                span.set_attribute("gen_ai.response.model", meta["model"])
            if "finish_reason" in meta:
                span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
            if "usage" in meta and meta["usage"]:
                usage = meta["usage"]
                if "prompt_tokens" in usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
                if "completion_tokens" in usage:
                    span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])
        print(f"    -> {reply.text[:60]}")


def run_agent():
    """Scenario: agent with tool calling via Haystack with prototype instrumentation."""
    print("  [agent] agent with tool calling (prototype)")
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool

    os.environ["OPENAI_API_KEY"] = "mock-key"
    request_model = "gpt-4o-mini"

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        function=get_weather,
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for",
                }
            },
            "required": ["location"],
        },
    )
    tool_definition = {
        "name": weather_tool.name,
        "description": weather_tool.description,
        "parameters": weather_tool.parameters,
    }

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([tool_definition]))

        agent = Agent(
            chat_generator=OpenAIChatGenerator(
                model=request_model,
                api_base_url=MOCK_BASE_URL,
            ),
            tools=[weather_tool],
            exit_conditions=["text"],
            max_agent_steps=3,
        )

        messages = [ChatMessage.from_user("What's the weather in Seattle?")]
        result = agent.run(messages=messages)
        reply = result.get("last_message")
        if reply is not None and getattr(reply, "meta", None):
            meta = reply.meta
            if "model" in meta:
                span.set_attribute("gen_ai.response.model", meta["model"])
            if "finish_reason" in meta:
                span.set_attribute("gen_ai.response.finish_reasons", [meta["finish_reason"]])
            if "usage" in meta and meta["usage"]:
                usage = meta["usage"]
                if "prompt_tokens" in usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
                if "completion_tokens" in usage:
                    span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])

        if reply is not None and getattr(reply, "text", None):
            print(f"    -> {reply.text[:60]}")
        else:
            print("    -> (no text reply)")


def main():
    print("=== Prototype: Haystack Conformance Test ===")

    # Pre-load slow haystack modules before connecting OTel to weaver,
    # otherwise weaver's inactivity timeout fires during the long import.
    import haystack  # noqa: F401

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    run_chat()
    run_agent()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
