"""Conformance test: OpenLLMetry (Traceloop) LangChain instrumentation."""

from common import MOCK_BASE_URL, run


def instrument():
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    LangchainInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OpenLLMetry: LangChain Conformance Test",
        instrument,
        openai_api_base=MOCK_BASE_URL,
        openai_api_key="mock-key",
    )
