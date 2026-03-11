"""Conformance test: OpenAI Agents SDK native tracing bridged to OTel.

The OpenAI Agents SDK has its own tracing system (Trace/Span) that is
independent of OpenTelemetry.  This test installs a custom TracingExporter
that converts finished Agents SDK spans into OTel spans so they are
exported via the standard OTLP pipeline.

Exercises: agent_run.
"""

import asyncio
import os

from opentelemetry import trace
from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan

from otel_setup import setup_otel, flush_and_shutdown

from agents.tracing import (
    set_trace_processors,
    Span as AgentSpan,
    AgentSpanData,
    GenerationSpanData,
    FunctionSpanData,
    HandoffSpanData,
    ResponseSpanData,
    GuardrailSpanData,
    CustomSpanData,
)
from agents.tracing.processors import TracingExporter, BatchTraceProcessor

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


# ---------------------------------------------------------------------------
# Bridge: Agents SDK tracing  ->  OpenTelemetry
# ---------------------------------------------------------------------------

class OTelBridgeExporter(TracingExporter):
    """Converts completed Agents SDK spans into OTel spans."""

    def __init__(self):
        self._tracer = trace.get_tracer("openai.agents")

    def export(self, items):
        for item in items:
            if isinstance(item, AgentSpan):
                self._export_span(item)

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _to_ns(dt):
        """ISO-format string or datetime -> nanoseconds since epoch (or None)."""
        if dt is None:
            return None
        if isinstance(dt, str):
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(dt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9)

    @staticmethod
    def _span_name(span_data):
        """Derive a human-readable OTel span name from the span_data type."""
        if isinstance(span_data, AgentSpanData):
            return f"Agent: {span_data.name}" if span_data.name else "Agent"
        if isinstance(span_data, GenerationSpanData):
            return "Generation"
        if isinstance(span_data, FunctionSpanData):
            return f"Function: {span_data.name}" if span_data.name else "Function"
        if isinstance(span_data, HandoffSpanData):
            return "Handoff"
        if isinstance(span_data, ResponseSpanData):
            return "Response"
        if isinstance(span_data, GuardrailSpanData):
            return f"Guardrail: {span_data.name}" if span_data.name else "Guardrail"
        if isinstance(span_data, CustomSpanData):
            return span_data.name or "Custom"
        return type(span_data).__name__

    @staticmethod
    def _parse_hex_id(raw_id, bits=128):
        """Strip common prefixes (trace_, span_) and parse as hex int."""
        if raw_id is None:
            return 0
        cleaned = raw_id
        for prefix in ("trace_", "span_"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        mask = (1 << bits) - 1
        return int(cleaned, 16) & mask

    def _build_parent_ctx(self, agent_span):
        """Build an OTel Context that carries the parent span reference."""
        trace_id = self._parse_hex_id(agent_span.trace_id, bits=128)
        if not agent_span.parent_id:
            parent_span_ctx = SpanContext(
                trace_id=trace_id,
                span_id=0,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
        else:
            parent_span_id = self._parse_hex_id(agent_span.parent_id, bits=64)
            parent_span_ctx = SpanContext(
                trace_id=trace_id,
                span_id=parent_span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
        return trace.set_span_in_context(NonRecordingSpan(parent_span_ctx))

    @staticmethod
    def _collect_attributes(span_data):
        """Return a dict of OTel span attributes derived from *span_data*."""
        attrs = {"gen_ai.system": "openai"}

        if isinstance(span_data, AgentSpanData):
            if span_data.name:
                attrs["gen_ai.agent.name"] = span_data.name
            if span_data.tools:
                attrs["gen_ai.agent.tools"] = str(span_data.tools)
            if span_data.output_type:
                attrs["gen_ai.agent.output_type"] = span_data.output_type

        elif isinstance(span_data, GenerationSpanData):
            if span_data.model:
                attrs["gen_ai.request.model"] = span_data.model
            if span_data.model_config:
                attrs["gen_ai.request.model_config"] = str(span_data.model_config)
            if span_data.usage:
                usage = span_data.usage
                if hasattr(usage, "input_tokens") and usage.input_tokens is not None:
                    attrs["gen_ai.usage.input_tokens"] = usage.input_tokens
                if hasattr(usage, "output_tokens") and usage.output_tokens is not None:
                    attrs["gen_ai.usage.output_tokens"] = usage.output_tokens

        elif isinstance(span_data, FunctionSpanData):
            if span_data.name:
                attrs["gen_ai.tool.name"] = span_data.name
            if span_data.input:
                attrs["gen_ai.tool.input"] = str(span_data.input)[:256]
            if span_data.output:
                attrs["gen_ai.tool.output"] = str(span_data.output)[:256]

        elif isinstance(span_data, HandoffSpanData):
            if span_data.from_agent:
                attrs["gen_ai.handoff.from_agent"] = span_data.from_agent
            if span_data.to_agent:
                attrs["gen_ai.handoff.to_agent"] = span_data.to_agent

        elif isinstance(span_data, GuardrailSpanData):
            if span_data.name:
                attrs["gen_ai.guardrail.name"] = span_data.name
            attrs["gen_ai.guardrail.triggered"] = span_data.triggered

        elif isinstance(span_data, CustomSpanData):
            if span_data.name:
                attrs["gen_ai.custom.name"] = span_data.name
            if span_data.data:
                attrs["gen_ai.custom.data"] = str(span_data.data)[:256]

        return attrs

    def _export_span(self, agent_span):
        start_ns = self._to_ns(agent_span.started_at)
        end_ns = self._to_ns(agent_span.ended_at)

        parent_ctx = self._build_parent_ctx(agent_span)
        name = self._span_name(agent_span.span_data)
        attributes = self._collect_attributes(agent_span.span_data)

        otel_span = self._tracer.start_span(
            name=name,
            context=parent_ctx,
            start_time=start_ns,
            attributes=attributes,
        )

        if agent_span.error:
            otel_span.set_status(
                trace.StatusCode.ERROR, str(agent_span.error)
            )
            otel_span.record_exception(
                Exception(str(agent_span.error))
            )

        otel_span.end(end_time=end_ns if end_ns else None)


# ---------------------------------------------------------------------------
# Agent scenario
# ---------------------------------------------------------------------------

async def run_agent():
    """Run a simple agent with the OpenAI Agents SDK."""
    from agents import Agent, Runner
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    import openai

    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    model = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client)

    agent = Agent(
        name="test-agent",
        instructions="You are a helpful assistant.",
        model=model,
    )

    print("  [agent_run] basic agent execution")
    result = await Runner.run(agent, "Say hello.")
    print(f"    -> {str(result.final_output)[:60]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Native: OpenAI Agents Conformance Test ===")

    tp, lp, mp = setup_otel()

    # Replace default Agents SDK processors with our OTel bridge.
    # Use a short schedule_delay so spans flush quickly in the test.
    exporter = OTelBridgeExporter()
    processor = BatchTraceProcessor(exporter, schedule_delay=0.5)
    set_trace_processors([processor])

    asyncio.run(run_agent())

    # Flush Agents SDK processor first so all spans reach OTel,
    # then flush OTel exporters.
    processor.force_flush()
    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
