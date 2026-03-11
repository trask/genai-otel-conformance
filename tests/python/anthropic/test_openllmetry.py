"""Conformance test: OpenLLMetry (Traceloop) Anthropic instrumentation."""

from common import run


def instrument():
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
    AnthropicInstrumentor().instrument()


if __name__ == "__main__":
    run("OpenLLMetry: Anthropic Conformance Test", instrument)
