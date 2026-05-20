"""Conformance test: OpenInference (Arize) Anthropic instrumentation."""

from common import run


def instrument():
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from openinference.instrumentation.config import TraceConfig

    AnthropicInstrumentor().instrument(config=TraceConfig(enable_genai_semconv=True))


if __name__ == "__main__":
    run("OpenInference: Anthropic Conformance Test", instrument)
