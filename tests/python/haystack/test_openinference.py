"""Conformance test: OpenInference (Arize) Haystack instrumentation.

Exercises: chat, agent via Haystack pipeline
against a mock OpenAI server, with the OpenInference Haystack instrumentation.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def instrument():
    from openinference.instrumentation.haystack import HaystackInstrumentor
    from openinference.instrumentation.config import TraceConfig

    HaystackInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


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
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool

    os.environ["OPENAI_API_KEY"] = "mock-key"

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

    agent = Agent(
        chat_generator=OpenAIChatGenerator(
            model="gpt-4o-mini",
            api_base_url=MOCK_BASE_URL,
        ),
        tools=[weather_tool],
        exit_conditions=["text"],
        max_agent_steps=3,
    )

    messages = [ChatMessage.from_user("What's the weather in Seattle?")]
    result = agent.run(messages=messages)
    last = result.get("last_message")
    if last and last.text:
        print(f"    -> {last.text[:60]}")
    else:
        print(f"    -> (no text reply)")


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
