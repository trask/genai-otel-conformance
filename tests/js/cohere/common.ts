/**
 * Shared test infrastructure for Cohere JS conformance tests.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import type { CohereClient } from "cohere-ai";

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

  const logExporter = new OTLPLogExporter({ url: OTLP_ENDPOINT });
  const loggerProvider = new LoggerProvider({ processors: [new BatchLogRecordProcessor(logExporter)] });

  return { provider, meterProvider, loggerProvider };
}

export async function runChat(client: any) {
  console.log("  [chat] basic chat completion");
  const resp = await client.chat({
    model: "command-r-plus",
    message: "Say hello.",
  });
  console.log(`    -> ${resp.text.slice(0, 60)}`);
}

export async function runEmbeddings(client: any) {
  console.log("  [embeddings] embedding generation");
  const resp = await client.embed({
    model: "embed-english-v3.0",
    texts: ["Hello, world!"],
    inputType: "search_document",
  });
  const embeddings = resp.embeddings as number[][];
  console.log(`    -> embedding dim: ${embeddings[0].length}`);
}

export async function run(title: string, instrumentFn: (cohereModule: any) => void) {
  console.log(`=== ${title} ===`);

  const { provider, meterProvider, loggerProvider } = setupOtel();

  // Import cohere-ai and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const cohereModule = await import("cohere-ai");
  instrumentFn(cohereModule);

  const { CohereClient } = cohereModule;
  const client = new CohereClient({ token: "mock-key", baseUrl: MOCK_BASE_URL });

  await runChat(client);
  await runEmbeddings(client);

  console.log("Flushing telemetry...");
  await provider.forceFlush();
  await meterProvider.forceFlush();
  await loggerProvider.forceFlush();
  await provider.shutdown();
  await meterProvider.shutdown();
  await loggerProvider.shutdown();
  console.log("Done.");
}
