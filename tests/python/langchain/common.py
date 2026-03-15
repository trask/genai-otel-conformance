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

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    try:
        # langchain >= 1.0: LangGraph-based create_agent
        from langchain.agents import create_agent
        agent = create_agent(llm, tools=[get_weather])
        result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Seattle?"}]})
        msgs = result.get("messages", [])
        output = msgs[-1].content if msgs else ""
        print(f"    -> {str(output)[:60]}")
    except ImportError:
        # langchain < 1.0: legacy AgentExecutor
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, [get_weather], prompt)
        executor = AgentExecutor(agent=agent, tools=[get_weather])
        result = executor.invoke({"input": "What's the weather in Seattle?"})
        print(f"    -> {str(result.get('output', ''))[:60]}")


def run_embeddings():
    """Scenario: LangChain embedding generation."""
    print("  [embeddings] embedding generation")
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base=MOCK_BASE_URL,
        openai_api_key="mock-key",
    )
    result = embeddings.embed_query("Hello, world!")
    print(f"    -> embedding dim: {len(result)}")


def run_retrieval():
    """Scenario: LangChain retrieval from in-memory vector store."""
    print("  [retrieval] vector store retrieval")
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base=MOCK_BASE_URL,
        openai_api_key="mock-key",
    )
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_texts(["The weather in Seattle is rainy."])
    retriever = vectorstore.as_retriever()
    results = retriever.invoke("What's the weather?")
    print(f"    -> retrieved {len(results)} document(s)")


def run(title, instrument_fn, **llm_kwargs):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", **llm_kwargs)

    run_chat(llm)
    run_chat_streaming(llm)
    try:
        run_agent(llm)
    except Exception as e:
        print(f"    WARNING: agent failed: {e}")
    run_embeddings()
    try:
        run_retrieval()
    except Exception as e:
        print(f"    WARNING: retrieval failed: {e}")

    flush_and_shutdown(tp, lp, mp)
