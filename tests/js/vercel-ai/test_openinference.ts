/**
 * Conformance test: Vercel AI SDK with OpenInference instrumentation.
 *
 * Exercises: chat, chat_streaming
 * against a mock OpenAI server, using @arizeai/openinference-vercel
 * to transform Vercel AI SDK telemetry spans into OpenInference semantic conventions.
 *
 * The OpenInferenceSimpleSpanProcessor wraps the OTLP exporter and converts
 * Vercel AI SDK's native spans (emitted via experimental_telemetry) into
 * OpenInference-compatible spans before export.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import { OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-vercel";
import { generateText, streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";
const OTLP_ENDPOINT = process.env.OTEL_EXPORTER_OTLP_ENDPOINT!;

function setupOtel() {
  const traceExporter = new OTLPTraceExporter({ url: OTLP_ENDPOINT });
  const provider = new NodeTracerProvider({
    spanProcessors: [
      new OpenInferenceSimpleSpanProcessor({
        exporter: traceExporter,
      }),
    ],
  });
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

async function runChat(openai: ReturnType<typeof createOpenAI>) {
  console.log("  [chat] basic chat completion");
  const { text } = await generateText({
    model: openai.chat("gpt-4o-mini"),
    prompt: "Say hello.",
    experimental_telemetry: { isEnabled: true },
  });
  console.log(`    -> ${text.slice(0, 60)}`);
}

async function runChatStreaming(openai: ReturnType<typeof createOpenAI>) {
  console.log("  [chat_streaming] streaming chat completion");
  const result = await streamText({
    model: openai.chat("gpt-4o-mini"),
    prompt: "Tell me a joke.",
    experimental_telemetry: { isEnabled: true },
  });

  let text = "";
  for await (const chunk of result.textStream) {
    text += chunk;
  }
  console.log(`    -> ${text.slice(0, 60)}`);
}

async function main() {
  console.log("=== OpenInference: Vercel AI SDK Conformance Test ===");

  const { provider, meterProvider, loggerProvider } = setupOtel();

  const openai = createOpenAI({
    baseURL: MOCK_BASE_URL,
    apiKey: "mock-key",
  });

  await runChat(openai);
  await runChatStreaming(openai);

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
