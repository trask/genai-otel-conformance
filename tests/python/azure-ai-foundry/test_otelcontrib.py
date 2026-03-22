"""Conformance test: Azure AI Foundry Agent native OTel instrumentation.

Exercises: invoke_agent (create agent, thread, message, run)
against a mock server, with the Azure SDK native OTel tracing.
"""

from common import run, run_invoke_agent


def instrument():
    # Azure AI Foundry agents use azure-core-tracing-opentelemetry for native OTel.
    # No separate instrumentor needed—azure-core picks up OTel automatically
    # when azure_settings.tracing_implementation = "opentelemetry" is set.
    pass


if __name__ == "__main__":
    run(
        "Native: Azure AI Foundry Agent Conformance Test",
        instrument,
        [run_invoke_agent],
    )
