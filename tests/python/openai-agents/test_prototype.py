"""Conformance test: prototype instrumentation for OpenAI Agents.

Exercises: agent run with tool calling
against a mock OpenAI server, with manual OTel spans.
"""

import asyncio
import os
from urllib.parse import urlparse

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


async def run_agent():
    """Run a simple agent with the OpenAI Agents SDK, with manual spans."""
    from agents import Agent, Runner, function_tool
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    import openai

    @function_tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    model = OpenAIChatCompletionsModel(model=request_model, openai_client=client)

    agent = Agent(
        name="test-agent",
        instructions="You are a helpful assistant.",
        model=model,
        tools=[get_weather],
    )

    print("  [agent_run] agent with tool calling (prototype)")
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.response.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        result = await Runner.run(agent, "What's the weather in Seattle?")
        usage = result.context_wrapper.usage
        if usage.total_tokens:
            span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        response_id = result.last_response_id
        if not response_id:
            for item in result.new_items:
                provider_data = getattr(item.raw_item, "provider_data", None)
                if isinstance(provider_data, dict) and provider_data.get("response_id"):
                    response_id = provider_data["response_id"]
                    break
        if response_id:
            span.set_attribute("gen_ai.response.id", response_id)
        print(f"    -> {str(result.final_output)[:60]}")


def main():
    print("=== Prototype: OpenAI Agents Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    asyncio.run(run_agent())

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
