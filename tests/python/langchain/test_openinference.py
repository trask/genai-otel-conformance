"""Conformance test: OpenInference (Arize) LangChain instrumentation."""

from common import MOCK_BASE_URL, run


def instrument():
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.config import TraceConfig

    LangChainInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run(
        "OpenInference: LangChain Conformance Test",
        instrument,
        openai_api_base=MOCK_BASE_URL,
        openai_api_key="mock-key",
    )
