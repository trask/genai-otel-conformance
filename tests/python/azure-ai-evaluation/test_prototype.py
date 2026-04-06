"""Conformance test: prototype instrumentation for Azure AI Evaluation."""

import os

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_evaluation():
    """Scenario: evaluation result event via Azure AI Evaluation."""
    from azure.ai.evaluation import OpenAIModelConfiguration, RelevanceEvaluator

    print("  [evaluate] Azure AI Evaluation relevance event")

    query = "What is the capital of France?"
    response = "Paris is the capital of France."
    model_config = OpenAIModelConfiguration(
        type="openai",
        api_key="mock-key",
        model="gpt-4o-mini",
        base_url=MOCK_BASE_URL,
    )
    evaluator = RelevanceEvaluator(model_config=model_config)
    evaluation_name = "relevance"

    with _prototype_tracer.start_as_current_span("prototype.evaluation", kind=SpanKind.INTERNAL) as span:
        try:
            result = evaluator(query=query, response=response)
            score = float(result["relevance"])
            score_label = str(result["relevance_result"])

            get_logger_provider().get_logger("gen_ai.evaluation.prototype").emit(
                event_name="gen_ai.evaluation.result",
                body="Evaluation result",
                attributes={
                    "gen_ai.evaluation.explanation": str(result["relevance_reason"]),
                    "gen_ai.evaluation.name": evaluation_name,
                    "gen_ai.evaluation.score.label": score_label,
                    "gen_ai.evaluation.score.value": score,
                },
            )

            print(f"    -> score: {score}")
            print(f"    -> label: {score_label}")
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__qualname__)
            raise


def main():
    print("=== Prototype: Azure AI Evaluation Conformance Test ===")

    tp, lp, mp = setup_otel()

    run_evaluation()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()