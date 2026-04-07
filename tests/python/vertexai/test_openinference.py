"""Conformance test: OpenInference (Arize) Vertex AI instrumentation.

Uses openinference-instrumentation-vertexai to instrument the
Vertex AI SDK.  Exercises: chat, chat_streaming.
"""

import os
import warnings

from opentelemetry import trace

from otel_setup import setup_otel, flush_and_shutdown

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

# The Vertex AI gapic REST transport defaults to HTTPS. Monkey-patch the
# transport to use plain HTTP so we can talk to the local mock server.
from google.cloud.aiplatform_v1.services.prediction_service.transports.rest import (  # noqa: E402
    PredictionServiceRestTransport,
)

_original_rest_init = PredictionServiceRestTransport.__init__


def _patched_rest_init(self, **kwargs):
    kwargs.setdefault("url_scheme", "http")
    return _original_rest_init(self, **kwargs)


PredictionServiceRestTransport.__init__ = _patched_rest_init


def _mock_host():
    """Return host:port from MOCK_BASE_URL (strip scheme)."""
    return MOCK_BASE_URL.replace("http://", "").replace("https://", "")


def instrument():
    from openinference.instrumentation.vertexai import VertexAIInstrumentor
    VertexAIInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())


def _init_vertexai():
    """Initialize Vertex AI SDK pointing at the mock server."""
    import vertexai
    from google.auth.credentials import AnonymousCredentials

    vertexai.init(
        project="test-project",
        location="us-central1",
        credentials=AnonymousCredentials(),
        api_endpoint=_mock_host(),
        api_transport="rest",
    )


def run_chat():
    """Scenario: basic chat completion via Vertex AI."""
    from vertexai.generative_models import GenerativeModel

    print("  [chat] basic chat completion via Vertex AI")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say hello.")
    print(f"    -> {response.text[:60]}")


def run_chat_streaming():
    """Scenario: streaming chat completion via Vertex AI."""
    from vertexai.generative_models import GenerativeModel

    print("  [chat_streaming] streaming chat completion via Vertex AI")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = GenerativeModel("gemini-2.0-flash")
        text = ""
        for chunk in model.generate_content("Tell me a joke.", stream=True):
            text += chunk.text
    print(f"    -> {text[:60]}")


def run_chat_tool_call():
    """Scenario: chat with tool calling via Vertex AI."""
    from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Tool

    print("  [chat_tool_call] chat with tool calling via Vertex AI")
    get_weather_func = FunctionDeclaration(
        name="get_weather",
        description="Get the current weather",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    )
    tool = Tool(function_declarations=[get_weather_func])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            "What's the weather in Seattle?",
            tools=[tool],
        )
    part = response.candidates[0].content.parts[0]
    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
        print(f"    -> tool_call: {part.function_call.name}")
    else:
        print(f"    -> {response.text[:60]}")


def main():
    print("=== OpenInference: Vertex AI Conformance Test ===")

    tp, lp, mp = setup_otel()
    instrument()
    _init_vertexai()

    run_chat()
    run_chat_streaming()
    run_chat_tool_call()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
