/**
 * Shared test infrastructure for Anthropic JS conformance tests.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { metrics } from "@opentelemetry/api";
import { logs } from "@opentelemetry/api-logs";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import type Anthropic from "@anthropic-ai/sdk";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;
const OTLP_ENDPOINT = process.env.OTEL_EXPORTER_OTLP_ENDPOINT!;

export function setupOtel() {
  const traceExporter = new OTLPTraceExporter({ url: OTLP_ENDPOINT });
  const provider = new NodeTracerProvider({ spanProcessors: [new BatchSpanProcessor(traceExporter)] });
  provider.register();

  const metricReader = new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({ url: OTLP_ENDPOINT }),
    exportIntervalMillis: 5000,
  });
  const meterProvider = new MeterProvider({ readers: [metricReader] });
  metrics.setGlobalMeterProvider(meterProvider);

  const logExporter = new OTLPLogExporter({ url: OTLP_ENDPOINT });
  const loggerProvider = new LoggerProvider({ processors: [new BatchLogRecordProcessor(logExporter)] });
  logs.setGlobalLoggerProvider(loggerProvider);

  return { provider, meterProvider, loggerProvider };
}

export async function runChat(client: Anthropic) {
  console.log("  [chat] basic chat completion");
  const resp = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 100,
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.content[0].text.slice(0, 60)}`);
}

export async function runChatStreaming(client: Anthropic) {
  console.log("  [chat_streaming] streaming chat completion");
  const stream = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 100,
    messages: [{ role: "user", content: "Tell me a joke." }],
    stream: true,
  });
  let text = "";
  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      text += event.delta.text;
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);
}

export async function run(title: string, instrumentFn: (anthropicModule: any) => void) {
  console.log(`=== ${title} ===`);

  const { provider, meterProvider, loggerProvider } = setupOtel();

  // Import @anthropic-ai/sdk and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const anthropicModule = await import("@anthropic-ai/sdk");
  instrumentFn(anthropicModule);

  const Anthropic = anthropicModule.default;
  const client = new Anthropic({ baseURL: MOCK_BASE_URL, apiKey: "mock-key" });

  await runChat(client);
  await runChatStreaming(client);

  console.log("Flushing telemetry...");
  await provider.forceFlush();
  await meterProvider.forceFlush();
  await loggerProvider.forceFlush();
  await provider.shutdown();
  await meterProvider.shutdown();
  await loggerProvider.shutdown();
  console.log("Done.");
}
