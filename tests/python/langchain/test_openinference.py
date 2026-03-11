"""Conformance test: OpenInference (Arize) LangChain instrumentation."""

from opentelemetry import trace

from common import MOCK_BASE_URL, run


def instrument():
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run(
        "OpenInference: LangChain Conformance Test",
        instrument,
        openai_api_base=MOCK_BASE_URL,
        openai_api_key="mock-key",
    )
