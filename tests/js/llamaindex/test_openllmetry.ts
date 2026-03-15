/**
 * Conformance test: OpenLLMetry (Traceloop) JS LlamaIndex instrumentation.
 *
 * Exercises: chat
 * against a mock OpenAI server.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import { LlamaIndexInstrumentation } from "@traceloop/instrumentation-llamaindex";
import type { OpenAI } from "@llamaindex/openai";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";
const OTLP_ENDPOINT = process.env.OTEL_EXPORTER_OTLP_ENDPOINT!;

function setupOtel() {
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

  return { provider, meterProvider, loggerProvider };
}

async function runChat(llm: OpenAI) {
  console.log("  [chat] basic chat completion");
  const resp = await llm.chat({
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.message.content.toString().slice(0, 60)}`);
}

async function runEmbeddings(OpenAIEmbedding: any) {
  console.log("  [embeddings] embedding generation");
  const embed = new OpenAIEmbedding({
    model: "text-embedding-3-small",
    apiKey: "mock-key",
    additionalSessionOptions: { baseURL: MOCK_BASE_URL },
  });
  const result = await embed.getTextEmbedding("Hello, world!");
  console.log(`    -> embedding dim: ${result.length}`);
}

async function main() {
  console.log("=== OpenLLMetry JS: LlamaIndex Conformance Test ===");

  const { provider, meterProvider, loggerProvider } = setupOtel();

  // Dynamic import AFTER OTel setup so manuallyInstrument can patch prototypes.
  // tsx (ESM) does not support require-in-the-middle, so enable() does not work.
  const llamaindexModule = await import("llamaindex");
  const openaiModule = await import("@llamaindex/openai");

  const instrumentation = new LlamaIndexInstrumentation();
  // manuallyInstrument patches the main llamaindex module (RetrieverQueryEngine, etc.)
  instrumentation.manuallyInstrument(llamaindexModule as any);

  const { Settings } = llamaindexModule;
  const { OpenAI, OpenAIEmbedding } = openaiModule;

  Settings.llm = new OpenAI({
    model: "gpt-4o-mini",
    additionalSessionOptions: { baseURL: MOCK_BASE_URL },
    apiKey: "mock-key",
  });

  const llm = new OpenAI({
    model: "gpt-4o-mini",
    additionalSessionOptions: { baseURL: MOCK_BASE_URL },
    apiKey: "mock-key",
  });

  await runChat(llm);
  await runEmbeddings(OpenAIEmbedding);

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
