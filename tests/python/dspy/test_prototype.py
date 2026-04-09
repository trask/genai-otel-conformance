"""Conformance test: prototype instrumentation for DSPy.

Exercises: chat via DSPy LM and evaluation result events
against a mock OpenAI server, with manual OTel spans.
"""

import json
import os

import dspy
from opentelemetry import trace
from opentelemetry._logs import get_logger_provider
from opentelemetry.trace import SpanKind, StatusCode

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"
MODEL_KEY = "openai/gpt-4o-mini"

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat():
    """Scenario: basic chat via DSPy LM with prototype instrumentation."""
    print("  [chat] basic chat via DSPy LM (prototype)")

    request_model = "gpt-4o-mini"
    lm_model = f"openai/{request_model}"
    prompt_text = "Say hello."
    lm = dspy.LM(
        model=lm_model,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        cache=False,
    )
    dspy.configure(lm=lm)

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.response.model", request_model)
        result = lm(prompt_text)
        history_entry = lm.history[-1] if lm.history else None
        if history_entry is not None:
            request_messages = history_entry.get("messages") or []
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps([
                    {
                        "role": message["role"],
                        "parts": [{"type": "text", "content": message["content"]}],
                    }
                    for message in request_messages
                    if isinstance(message.get("content"), str)
                ]),
            )
            response = history_entry.get("response")
            if response is not None:
                if getattr(response, "id", None):
                    span.set_attribute("gen_ai.response.id", response.id)
                finish_reasons = [
                    str(choice.finish_reason).lower()
                    for choice in response.choices
                    if getattr(choice, "finish_reason", None)
                ]
                if finish_reasons:
                    span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
                span.set_attribute(
                    "gen_ai.output.messages",
                    json.dumps([
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": choice.message.content}],
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response.choices
                        if getattr(choice.message, "content", None)
                    ]),
                )
        usage = lm.history[-1].get("usage", {}) if lm.history else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
        if completion_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.response.model": request_model,
        }
        if history_entry is not None:
            if request_messages:
                event_attrs["gen_ai.input.messages"] = json.dumps([
                    {"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]}
                    for m in request_messages if isinstance(m.get("content"), str)
                ])
            response = history_entry.get("response")
            if response is not None:
                if getattr(response, "id", None):
                    event_attrs["gen_ai.response.id"] = response.id
                finish_reasons = [
                    str(choice.finish_reason).lower()
                    for choice in response.choices
                    if getattr(choice, "finish_reason", None)
                ]
                if finish_reasons:
                    event_attrs["gen_ai.response.finish_reasons"] = finish_reasons
                event_attrs["gen_ai.output.messages"] = json.dumps([
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": choice.message.content}],
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                    if getattr(choice.message, "content", None)
                ])
        if prompt_tokens is not None:
            event_attrs["gen_ai.usage.input_tokens"] = prompt_tokens
        if completion_tokens is not None:
            event_attrs["gen_ai.usage.output_tokens"] = completion_tokens
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {str(result)[:60]}")


def run_tool_call():
    """Scenario: tool calling via DSPy ReAct with prototype instrumentation."""
    print("  [chat_tool_call] tool calling via DSPy ReAct (prototype)")

    request_model = "gpt-4o-mini"
    lm_model = f"openai/{request_model}"
    prompt_text = "What's the weather in Seattle?"
    lm = dspy.LM(
        model=lm_model,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        cache=False,
    )
    dspy.configure(lm=lm)
    messages = [{"role": "user", "content": prompt_text}]
    tool_definition = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "type": "object",
        },
    }
    request_tool = {
        "type": "function",
        "function": {
            "name": tool_definition["name"],
            "description": tool_definition["description"],
            "parameters": tool_definition["parameters"],
        },
    }

    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([tool_definition]))
        result = lm(
            messages=messages,
            tools=[request_tool],
        )
        history_entry = lm.history[-1] if lm.history else None
        if history_entry is not None:
            request_messages = history_entry.get("messages") or []
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps([
                    {
                        "role": message["role"],
                        "parts": [{"type": "text", "content": message["content"]}],
                    }
                    for message in request_messages
                    if isinstance(message.get("content"), str)
                ]),
            )
            response = history_entry.get("response")
            if response is not None:
                if getattr(response, "id", None):
                    span.set_attribute("gen_ai.response.id", response.id)
                finish_reasons = [
                    str(choice.finish_reason).lower()
                    for choice in response.choices
                    if getattr(choice, "finish_reason", None)
                ]
                if finish_reasons:
                    span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
                span.set_attribute(
                    "gen_ai.output.messages",
                    json.dumps([
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": choice.message.content}],
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response.choices
                        if getattr(choice.message, "content", None)
                    ]),
                )
        usage = lm.history[-1].get("usage", {}) if lm.history else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
        if completion_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
        print(f"    -> {str(result)[:60]}")


class EchoProgram(dspy.Module):
    def forward(self, question):
        result = dspy.settings.lm(question)
        text = result[0] if isinstance(result, list) else str(result)
        return dspy.Prediction(answer=text)


def contains_mock_response(_example, prediction, trace=None):
    del trace
    return float("mock response" in getattr(prediction, "answer", "").lower())


def run_evaluation():
    """Scenario: evaluation result event with prototype instrumentation."""
    print("  [evaluate] DSPy evaluation result event")

    lm = dspy.LM(
        model=MODEL_KEY,
        api_base=MOCK_BASE_URL,
        api_key="mock-key",
        cache=False,
    )
    dspy.configure(lm=lm, track_usage=True)

    devset = [
        dspy.Example(
            question="Say hello.",
            answer="This is a mock response from the conformance test server.",
        ).with_inputs("question"),
    ]

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=contains_mock_response,
        display_progress=False,
        display_table=False,
    )

    with _prototype_tracer.start_as_current_span("prototype.evaluation", kind=SpanKind.INTERNAL) as span:
        try:
            result = evaluate(EchoProgram())
            score = float(result.score)
            score_label = "pass" if score > 0 else "fail"
            explanation = (
                "The evaluated response contained the expected mock-response marker."
                if score > 0 else
                "The evaluated response did not contain the expected mock-response marker."
            )
            history_entry = lm.history[-1] if lm.history else None
            response = history_entry.get("response") if history_entry is not None else None
            response_id = getattr(response, "id", None)

            attributes = {
                "gen_ai.evaluation.name": contains_mock_response.__name__,
                "gen_ai.evaluation.score.label": score_label,
                "gen_ai.evaluation.score.value": score,
                "gen_ai.evaluation.explanation": explanation,
            }
            if response_id:
                attributes["gen_ai.response.id"] = response_id
            get_logger_provider().get_logger("gen_ai.evaluation.prototype").emit(
                event_name="gen_ai.evaluation.result",
                body="Evaluation result",
                attributes=attributes,
            )

            print(f"    -> score: {result.score}")
            print(f"    -> results: {len(result.results)}")
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__qualname__)
            raise


def main():
    print("=== Prototype: DSPy Conformance Test ===")

    tp, lp, mp = setup_otel()

    run_chat()
    run_tool_call()
    run_evaluation()

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
