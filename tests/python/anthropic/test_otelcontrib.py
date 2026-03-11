"""Conformance test: OTel contrib opentelemetry-instrumentation-anthropic."""

from common import run


def instrument():
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
    AnthropicInstrumentor().instrument()


if __name__ == "__main__":
    run("OTel Contrib: Anthropic Conformance Test", instrument)
