"""Shared test infrastructure for LangChain conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def run_chat(llm):
    """Scenario: basic LangChain invoke."""
    print("  [chat] basic chat completion")
    resp = llm.invoke("Say hello.")
    print(f"    -> {resp.content[:60]}")


def run_chat_streaming(llm):
    """Scenario: LangChain streaming."""
    print("  [chat_streaming] streaming chat completion")
    text = ""
    for chunk in llm.stream("Tell me a joke."):
        text += chunk.content
    print(f"    -> {text[:60]}")


def run_agent(llm):
    """Scenario: LangChain agent with tool calling."""
    print("  [agent] agent with tool calling")
    from langchain_core.tools import tool
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [get_weather], prompt)
    executor = AgentExecutor(agent=agent, tools=[get_weather])
    result = executor.invoke({"input": "What's the weather in Seattle?"})
    print(f"    -> {str(result.get('output', ''))[:60]}")


def run(title, instrument_fn, **llm_kwargs):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", **llm_kwargs)

    run_chat(llm)
    run_chat_streaming(llm)
    run_agent(llm)

    flush_and_shutdown(tp, lp, mp)
