/**
 * Conformance test: OTel Contrib (@opentelemetry/instrumentation-openai) JS OpenAI instrumentation.
 *
 * The @opentelemetry/instrumentation-openai package uses InstrumentationBase's
 * module hooks which don't work under tsx ESM. We work around this by:
 * 1. Setting up the tracer/meter providers on the instrumentation instance
 * 2. Calling init() to get module definitions and invoking patch() directly
 */

import { OpenAIInstrumentation } from "@opentelemetry/instrumentation-openai";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import * as api from "@opentelemetry/api";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";
const OTLP_ENDPOINT = process.env.OTEL_EXPORTER_OTLP_ENDPOINT!;

async function main() {
  console.log("=== OTel Contrib JS: OpenAI Conformance Test ===");

  // Set up OTel providers
  const traceExporter = new OTLPTraceExporter({ url: OTLP_ENDPOINT });
  const provider = new NodeTracerProvider({ spanProcessors: [new BatchSpanProcessor(traceExporter)] });
  provider.register();

  const metricReader = new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({ url: OTLP_ENDPOINT }),
    exportIntervalMillis: 5000,
  });
  const meterProvider = new MeterProvider({ readers: [metricReader] });

  const logExporter = new OTLPLogExporter({ url: OTLP_ENDPOINT });
  const loggerProvider = new LoggerProvider({ processors: [new BatchLogRecordProcessor(logExporter)] });

  // Create instrumentation and set providers so meter/tracer are available
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.setTracerProvider(provider);
  instrumentation.setMeterProvider(meterProvider);

  // Import openai then manually patch
  const openaiModule = await import("openai");
  const defs = (instrumentation as any).init();
  for (const def of defs) {
    if (typeof def.patch === "function") {
      def.patch(openaiModule);
    }
  }

  const OpenAI = openaiModule.default;
  const client = new OpenAI({ baseURL: MOCK_BASE_URL, apiKey: "mock-key" });

  // Scenario: basic chat
  console.log("  [chat] basic chat completion");
  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.choices[0].message.content?.slice(0, 60)}`);

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion");
  const stream = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Tell me a joke." }],
    stream: true,
  });
  let text = "";
  for await (const chunk of stream) {
    if (chunk.choices[0]?.delta?.content) {
      text += chunk.choices[0].delta.content;
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation");
  const embResp = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: "Hello, world!",
  });
  console.log(`    -> embedding dim: ${embResp.data[0].embedding.length}`);

  console.log("Flushing telemetry...");
  await provider.forceFlush();
  await meterProvider.forceFlush();
  await loggerProvider.forceFlush();
  await provider.shutdown();
  await meterProvider.shutdown();
  await loggerProvider.shutdown();
  console.log("Done.");
}

main().catch((e) => { console.error(e); process.exit(1); });
