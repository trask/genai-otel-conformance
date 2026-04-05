"""Conformance test: prototype instrumentation for Pydantic AI.

Exercises: chat via OpenAI client
against a mock OpenAI server, with manual OTel spans (no logfire/Agent.instrument_all).
"""

import json
import os
from urllib.parse import urlparse

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat via Pydantic AI Agent with prototype instrumentation."""
    print("  [chat] basic chat via Pydantic AI Agent (prototype)")
    from pydantic_ai import Agent
    from pydantic_ai.messages import TextPart
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    system_prompt = "You are a helpful assistant."
    prompt_text = "Say hello."
    model = OpenAIChatModel(request_model, provider=provider)
    agent = Agent(model, system_prompt=system_prompt)

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps([
                {"role": "system", "parts": [{"type": "text", "content": system_prompt}]},
                {"role": "user", "parts": [{"type": "text", "content": prompt_text}]},
            ]),
        )
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        result = agent.run_sync(prompt_text)
        if result.response.model_name:
            span.set_attribute("gen_ai.response.model", result.response.model_name)
        if result.response.provider_response_id:
            span.set_attribute("gen_ai.response.id", result.response.provider_response_id)
        if result.response.finish_reason is not None:
            span.set_attribute(
                "gen_ai.response.finish_reasons",
                [result.response.finish_reason],
            )
        output_parts = [
            {"type": "text", "content": part.content}
            for part in result.response.parts
            if isinstance(part, TextPart) and part.content
        ]
        if output_parts:
            span.set_attribute(
                "gen_ai.output.messages",
                json.dumps([
                    {
                        "role": "assistant",
                        "parts": output_parts,
                        **(
                            {"finish_reason": result.response.finish_reason}
                            if result.response.finish_reason is not None
                            else {}
                        ),
                    }
                ]),
            )
        usage = result.usage()
        if usage.total_tokens:
            span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.input.messages": json.dumps([
                {"role": "system", "parts": [{"type": "text", "content": system_prompt}]},
                {"role": "user", "parts": [{"type": "text", "content": prompt_text}]},
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": output_parts,
                    **({"finish_reason": result.response.finish_reason}
                       if result.response.finish_reason is not None else {}),
                }
            ]),
        }
        if result.response.model_name:
            event_attrs["gen_ai.response.model"] = result.response.model_name
        if result.response.provider_response_id:
            event_attrs["gen_ai.response.id"] = result.response.provider_response_id
        if result.response.finish_reason is not None:
            event_attrs["gen_ai.response.finish_reasons"] = [result.response.finish_reason]
        if usage.total_tokens:
            event_attrs["gen_ai.usage.input_tokens"] = usage.input_tokens
            event_attrs["gen_ai.usage.output_tokens"] = usage.output_tokens
        if endpoint.hostname:
            event_attrs["server.address"] = endpoint.hostname
        if endpoint.port is not None:
            event_attrs["server.port"] = endpoint.port
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {str(result.response)[:60]}")


def run_tool_call():
    """Scenario: tool calling via Pydantic AI Agent with prototype instrumentation."""
    print("  [chat_tool_call] tool calling via Pydantic AI Agent (prototype)")
    from pydantic_ai import Agent, Tool
    from pydantic_ai.messages import TextPart
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    system_prompt = "You are a helpful assistant."
    prompt_text = "What's the weather in Seattle?"
    model = OpenAIChatModel(request_model, provider=provider)

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    tools = [Tool(get_weather)]
    agent = Agent(model, system_prompt=system_prompt, tools=tools)

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps([
                {"role": "system", "parts": [{"type": "text", "content": system_prompt}]},
                {"role": "user", "parts": [{"type": "text", "content": prompt_text}]},
            ]),
        )
        span.set_attribute("gen_ai.tool.definitions", json.dumps([
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.function_schema.json_schema,
            }
            for t in tools
        ]))
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        result = agent.run_sync(prompt_text)
        if result.response.model_name:
            span.set_attribute("gen_ai.response.model", result.response.model_name)
        if result.response.provider_response_id:
            span.set_attribute("gen_ai.response.id", result.response.provider_response_id)
        if result.response.finish_reason is not None:
            span.set_attribute(
                "gen_ai.response.finish_reasons",
                [result.response.finish_reason],
            )
        output_parts = [
            {"type": "text", "content": part.content}
            for part in result.response.parts
            if isinstance(part, TextPart) and part.content
        ]
        if output_parts:
            span.set_attribute(
                "gen_ai.output.messages",
                json.dumps([
                    {
                        "role": "assistant",
                        "parts": output_parts,
                        **(
                            {"finish_reason": result.response.finish_reason}
                            if result.response.finish_reason is not None
                            else {}
                        ),
                    }
                ]),
            )
        usage = result.usage()
        if usage.total_tokens:
            span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        print(f"    -> {str(result.response)[:60]}")


def main():
    print("=== Prototype: Pydantic AI Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO logfire.configure() or Agent.instrument_all() – prototype instrumentation only

    run_chat()
    run_tool_call()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
