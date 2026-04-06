"""Conformance test: prototype instrumentation for Azure AI Evaluation."""

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import flush_and_shutdown, setup_otel

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_evaluation():
    """Scenario: evaluation result event via Azure AI Evaluation."""
    from azure.ai.evaluation import F1ScoreEvaluator

    print("  [evaluate] Azure AI Evaluation F1 score event")

    ground_truth = "Paris is the capital of France."
    response = "Paris is the capital of France."
    evaluator = F1ScoreEvaluator()
    evaluation_name = "f1_score"

    with _prototype_tracer.start_as_current_span("prototype.evaluation", kind=SpanKind.INTERNAL) as span:
        try:
            result = evaluator(response=response, ground_truth=ground_truth)
            score = float(result["f1_score"])
            score_label = str(result["f1_result"])

            get_logger_provider().get_logger("gen_ai.evaluation.prototype").emit(
                event_name="gen_ai.evaluation.result",
                body="Evaluation result",
                attributes={
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