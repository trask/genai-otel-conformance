"""Conformance test: OpenInference (Arize) Haystack instrumentation.

Exercises: chat, agent via Haystack pipeline
against a mock OpenAI server, with the OpenInference Haystack instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from openinference.instrumentation.haystack import HaystackInstrumentor
    HaystackInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


def run_chat():
    print("  [chat] basic chat via Haystack pipeline")
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    os.environ["OPENAI_API_KEY"] = "mock-key"

    generator = OpenAIChatGenerator(
        model="gpt-4o-mini",
        api_base_url=MOCK_BASE_URL,
    )

    messages = [ChatMessage.from_user("Say hello.")]
    result = generator.run(messages=messages)
    reply = result["replies"][0]
    print(f"    -> {reply.text[:60]}")


def run_agent():
    """Scenario: Haystack agent with tool calling."""
    print("  [agent] agent with tool calling")
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    os.environ["OPENAI_API_KEY"] = "mock-key"

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    # Try the Tool class (Haystack >=2.8), fall back to generation_kwargs
    try:
        from haystack.tools import Tool

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
        generator = OpenAIChatGenerator(
            model="gpt-4o-mini",
            api_base_url=MOCK_BASE_URL,
            tools=[weather_tool],
        )
    except ImportError:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        generator = OpenAIChatGenerator(
            model="gpt-4o-mini",
            api_base_url=MOCK_BASE_URL,
            generation_kwargs={"tools": tools},
        )

    messages = [ChatMessage.from_user("What's the weather in Seattle?")]

    # First call — should get tool call back
    result = generator.run(messages=messages)
    replies = result["replies"]

    if replies and replies[0].tool_calls:
        tool_call = replies[0].tool_calls[0]
        tool_result = get_weather(**tool_call.arguments)
        print(f"    -> tool called: {tool_call.tool_name}({tool_call.arguments})")

        # Send tool result back
        messages.append(replies[0])
        messages.append(ChatMessage.from_tool(tool_result, origin=tool_call))
        result = generator.run(messages=messages)
        reply = result["replies"][0]
        print(f"    -> {reply.text[:60]}")
    else:
        print(f"    -> {replies[0].text[:60]}")


def main():
    print("=== OpenInference: Haystack Conformance Test ===")

    # Pre-load slow haystack modules before connecting OTel to weaver,
    # otherwise weaver's inactivity timeout fires during the long import.
    import haystack  # noqa: F401

    tp, lp, mp = setup_otel()
    instrument()

    run_chat()
    run_agent()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
