/**
 * Shared test infrastructure for OpenAI JS conformance tests.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import type OpenAI from "openai";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";
const OTLP_ENDPOINT = process.env.OTEL_EXPORTER_OTLP_ENDPOINT!;

export function setupOtel() {
  const traceExporter = new OTLPTraceExporter({ url: OTLP_ENDPOINT });
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new BatchSpanProcessor(traceExporter));
  provider.register();

  const metricReader = new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({ url: OTLP_ENDPOINT }),
    exportIntervalMillis: 5000,
  });
  const meterProvider = new MeterProvider({ readers: [metricReader] });

  const logExporter = new OTLPLogExporter({ url: OTLP_ENDPOINT });
  const loggerProvider = new LoggerProvider();
  loggerProvider.addLogRecordProcessor(new BatchLogRecordProcessor(logExporter));

  return { provider, meterProvider, loggerProvider };
}

export async function runChat(client: OpenAI) {
  console.log("  [chat] basic chat completion");
  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.choices[0].message.content?.slice(0, 60)}`);
}

export async function runChatStreaming(client: OpenAI) {
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
}

export async function runEmbeddings(client: OpenAI) {
  console.log("  [embeddings] embedding generation");
  const resp = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: "Hello, world!",
  });
  console.log(`    -> embedding dim: ${resp.data[0].embedding.length}`);
}

export async function run(title: string, instrumentFn: (openaiModule: any) => void) {
  console.log(`=== ${title} ===`);

  const { provider, meterProvider, loggerProvider } = setupOtel();

  // Import openai and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const openaiModule = await import("openai");
  instrumentFn(openaiModule);

  const OpenAI = openaiModule.default;
  const client = new OpenAI({ baseURL: MOCK_BASE_URL, apiKey: "mock-key" });

  await runChat(client);
  await runChatStreaming(client);
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
