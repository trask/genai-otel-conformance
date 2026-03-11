"""Conformance test: OTel contrib opentelemetry-instrumentation-langchain."""

from common import MOCK_BASE_URL, run


def instrument():
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    LangchainInstrumentor().instrument()


if __name__ == "__main__":
    run(
        "OTel Contrib: LangChain Conformance Test",
        instrument,
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )
