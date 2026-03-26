"""Conformance test: reference instrumentation for Vertex AI.

Exercises: chat, chat_streaming
against a mock Vertex AI server, with manual OTel spans.
"""

import os
import warnings

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

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

_reference_tracer = trace.get_tracer("gen_ai.reference")


def _mock_host():
    """Return host:port from MOCK_BASE_URL (strip scheme)."""
    return MOCK_BASE_URL.replace("http://", "").replace("https://", "")


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
    """Scenario: basic chat completion with reference instrumentation."""
    from vertexai.generative_models import GenerativeModel

    print("  [chat] basic chat completion via Vertex AI (reference)")
    request_model = "gemini-2.0-flash"
    with _reference_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "vertex_ai")
        span.set_attribute("gen_ai.request.model", request_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = GenerativeModel(request_model)
            response = model.generate_content("Say hello.")
        response_model = response.to_dict().get("modelVersion")
        if response_model:
            span.set_attribute("gen_ai.response.model", response_model)
        else:
            span.set_attribute("gen_ai.response.model", request_model)
        if response.candidates and response.candidates[0].finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [str(response.candidates[0].finish_reason.name)])
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            if response.usage_metadata.prompt_token_count:
                span.set_attribute("gen_ai.usage.input_tokens", response.usage_metadata.prompt_token_count)
            if response.usage_metadata.candidates_token_count:
                span.set_attribute("gen_ai.usage.output_tokens", response.usage_metadata.candidates_token_count)
        print(f"    -> {response.text[:60]}")


def run_chat_streaming():
    """Scenario: streaming chat completion with reference instrumentation."""
    from vertexai.generative_models import GenerativeModel

    print("  [chat_streaming] streaming chat completion via Vertex AI (reference)")
    request_model = "gemini-2.0-flash"
    with _reference_tracer.start_as_current_span("chat gemini-2.0-flash") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "vertex_ai")
        span.set_attribute("gen_ai.request.model", request_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = GenerativeModel(request_model)
            text = ""
            last_chunk = None
            for chunk in model.generate_content("Tell me a joke.", stream=True):
                text += chunk.text
                last_chunk = chunk
        if last_chunk:
            response_model = last_chunk.to_dict().get("modelVersion")
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)
            else:
                span.set_attribute("gen_ai.response.model", request_model)
        if last_chunk and last_chunk.candidates and last_chunk.candidates[0].finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [str(last_chunk.candidates[0].finish_reason.name)])
        if last_chunk and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
            if last_chunk.usage_metadata.prompt_token_count:
                span.set_attribute("gen_ai.usage.input_tokens", last_chunk.usage_metadata.prompt_token_count)
            if last_chunk.usage_metadata.candidates_token_count:
                span.set_attribute("gen_ai.usage.output_tokens", last_chunk.usage_metadata.candidates_token_count)
        print(f"    -> {text[:60]}")


def main():
    print("=== Reference: Vertex AI Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – reference instrumentation only
    _init_vertexai()

    run_chat()
    run_chat_streaming()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
