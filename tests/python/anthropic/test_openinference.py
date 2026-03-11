"""Conformance test: OpenInference (Arize) Anthropic instrumentation."""

from opentelemetry import trace

from common import run


def instrument():
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    AnthropicInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


if __name__ == "__main__":
    run("OpenInference: Anthropic Conformance Test", instrument)
