"""Conformance test: OTel contrib Azure AI Inference instrumentation."""

from common import run, run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings


def instrument():
    from azure.ai.inference.tracing import AIInferenceInstrumentor
    AIInferenceInstrumentor().instrument(enable_content_recording=True)


if __name__ == "__main__":
    run(
        "OTel Contrib: Azure AI Inference Conformance Test",
        instrument,
        [run_chat, run_chat_streaming, run_chat_tool_call, run_embeddings],
    )
