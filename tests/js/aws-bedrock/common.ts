/**
 * Shared test infrastructure for AWS Bedrock JS conformance tests.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import type { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;
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

export async function runChat(client: BedrockRuntimeClient, ConverseCommand: any) {
  console.log("  [chat] basic chat completion");
  const resp = await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [
        {
          role: "user",
          content: [{ text: "Say hello." }],
        },
      ],
    })
  );
  console.log(`    -> ${resp.output?.message?.content?.[0]?.text?.slice(0, 60)}`);
}

export async function runChatStreaming(client: BedrockRuntimeClient, ConverseStreamCommand: any) {
  console.log("  [chat_streaming] streaming chat completion");
  const resp = await client.send(
    new ConverseStreamCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [
        {
          role: "user",
          content: [{ text: "Tell me a joke." }],
        },
      ],
    })
  );
  let text = "";
  if (resp.stream) {
    for await (const event of resp.stream) {
      if (event.contentBlockDelta?.delta?.text) {
        text += event.contentBlockDelta.delta.text;
      }
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);
}

export async function runEmbeddings(client: BedrockRuntimeClient, InvokeModelCommand: any) {
  console.log("  [embeddings] Bedrock Titan Embeddings");
  const resp = await client.send(
    new InvokeModelCommand({
      modelId: "amazon.titan-embed-text-v2:0",
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify({ inputText: "Hello, world!" }),
    })
  );
  const result = JSON.parse(new TextDecoder().decode(resp.body));
  console.log(`    -> embedding dim: ${result.embedding.length}`);
}

export async function run(title: string, instrumentFn: (bedrockModule: any) => void) {
  console.log(`=== ${title} ===`);

  const { provider, meterProvider, loggerProvider } = setupOtel();

  // Import AWS Bedrock SDK and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const bedrockModule = await import("@aws-sdk/client-bedrock-runtime");
  instrumentFn(bedrockModule);

  const { BedrockRuntimeClient, ConverseCommand, ConverseStreamCommand, InvokeModelCommand } = bedrockModule;
  const client = new BedrockRuntimeClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
  });

  await runChat(client, ConverseCommand);
  await runChatStreaming(client, ConverseStreamCommand);
  await runEmbeddings(client, InvokeModelCommand);

  console.log("Flushing telemetry...");
  await provider.forceFlush();
  await meterProvider.forceFlush();
  await loggerProvider.forceFlush();
  await provider.shutdown();
  await meterProvider.shutdown();
  await loggerProvider.shutdown();
  console.log("Done.");
}
