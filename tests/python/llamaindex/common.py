"""Shared test infrastructure for LlamaIndex conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


def run_chat(llm):
    """Scenario: basic LlamaIndex chat completion."""
    print("  [chat] basic chat completion")
    from llama_index.core.llms import ChatMessage, MessageRole

    resp = llm.chat([ChatMessage(role=MessageRole.USER, content="Say hello.")])
    print(f"    -> {str(resp)[:60]}")


def run_chat_streaming(llm):
    """Scenario: LlamaIndex streaming chat completion."""
    print("  [chat_streaming] streaming chat completion")
    from llama_index.core.llms import ChatMessage, MessageRole

    text = ""
    stream_resp = llm.stream_chat(
        [ChatMessage(role=MessageRole.USER, content="Tell me a joke.")]
    )
    for token in stream_resp:
        text += token.delta
    print(f"    -> {text[:60]}")


def run_agent(llm):
    """Scenario: LlamaIndex agent with tool calling."""
    print("  [agent] agent with tool calling")
    from llama_index.core.tools import FunctionTool

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny, 72°F"

    weather_tool = FunctionTool.from_defaults(fn=get_weather)

    try:
        from llama_index.core.agent import FunctionCallingAgent

        agent = FunctionCallingAgent.from_tools(
            tools=[weather_tool],
            llm=llm,
            verbose=False,
        )
    except ImportError:
        from llama_index.core.agent import ReActAgent

        agent = ReActAgent.from_tools(
            tools=[weather_tool],
            llm=llm,
            verbose=False,
        )

    response = agent.chat("What's the weather in Seattle?")
    print(f"    -> {str(response)[:60]}")


def run_embeddings():
    """Scenario: LlamaIndex embedding generation."""
    print("  [embeddings] embedding generation")
    from llama_index.embeddings.openai import OpenAIEmbedding
    embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )
    result = embed_model.get_text_embedding("Hello, world!")
    print(f"    -> embedding dim: {len(result)}")


def run(title, instrument_fn):
    """Run conformance test scenarios."""
    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    llm = LlamaOpenAI(
        model="gpt-4o-mini",
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
    )

    run_chat(llm)
    run_chat_streaming(llm)
    try:
        run_agent(llm)
    except Exception as e:
        print(f"    WARNING: agent failed: {e}")
    run_embeddings()

    flush_and_shutdown(tp, lp, mp)
