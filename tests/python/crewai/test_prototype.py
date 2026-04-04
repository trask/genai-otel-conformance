"""Conformance test: prototype instrumentation for CrewAI.

Exercises: agent task execution
against a mock OpenAI server, with manual OTel spans.
"""

import json
import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_crew():
    """Scenario: basic crew task execution with prototype instrumentation."""
    print("  [crew] basic crew task execution (prototype)")
    from crewai import Agent, Task, Crew
    from crewai.tools import tool

    request_model = "gpt-4o-mini"
    os.environ["OPENAI_API_KEY"] = "mock-key"
    os.environ["OPENAI_API_BASE"] = MOCK_BASE_URL
    os.environ["OPENAI_MODEL_NAME"] = request_model

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    tools = [get_weather]

    researcher = Agent(
        role="Researcher",
        goal="Find information",
        backstory="You are a helpful research assistant.",
        tools=tools,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description="What's the weather in Seattle?",
        expected_output="The current weather.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=False)

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        # CrewAI converts tools to OpenAI function-calling format before
        # passing them to litellm, so we mirror that shape here.
        span.set_attribute("gen_ai.tool.definitions", json.dumps([
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.func.__doc__,
                    "parameters": t.args_schema.model_json_schema(),
                },
            }
            for t in tools
        ]))
        result = crew.kickoff()
        print(f"    -> {str(result)[:60]}")


def main():
    print("=== Prototype: CrewAI Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    run_crew()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
