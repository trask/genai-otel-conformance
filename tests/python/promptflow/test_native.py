"""Conformance test: Promptflow native OTel instrumentation.

Exercises: chat via @trace-decorated function
against a mock OpenAI server, with Promptflow's built-in tracing.
"""

import os
import sys

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"
OTLP_ENDPOINT = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]


def setup_otel():
    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)))
    trace.set_tracer_provider(tp)
    lp = LoggerProvider()
    lp.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTLP_ENDPOINT, insecure=True)))
    set_logger_provider(lp)
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True),
        export_interval_millis=5000,
    )
    mp = MeterProvider(metric_readers=[reader])
    return tp, lp, mp


def run_chat():
    print("  [chat] basic chat via @trace decorator")
    from promptflow.tracing import trace as pf_trace
    import openai

    client = openai.OpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")

    @pf_trace
    def chat_completion(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    result = chat_completion("Say hello.")
    print(f"    -> {result[:60]}")


def main():
    print("=== Native: Promptflow Conformance Test ===")

    # Promptflow disables tracing by default (PF_DISABLE_TRACING defaults to "true")
    os.environ["PF_DISABLE_TRACING"] = "false"

    # start_trace() creates its own TracerProvider, so call it FIRST,
    # then add our OTLP exporter to the provider it created.
    from promptflow.tracing import start_trace
    start_trace()

    # Now get the active TracerProvider and add the OTLP exporter
    tp = trace.get_tracer_provider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)))

    lp = LoggerProvider()
    lp.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTLP_ENDPOINT, insecure=True)))
    set_logger_provider(lp)
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True),
        export_interval_millis=5000,
    )
    mp = MeterProvider(metric_readers=[reader])

    run_chat()

    print("Flushing telemetry...")
    tp.force_flush()
    lp.force_flush()
    mp.force_flush()
    tp.shutdown()
    lp.shutdown()
    mp.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
