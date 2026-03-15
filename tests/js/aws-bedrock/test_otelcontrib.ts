/**
 * Conformance test: OTel Contrib (@opentelemetry/instrumentation-aws-sdk) JS AWS Bedrock instrumentation.
 *
 * Uses registerInstrumentations + dynamic import() so that require-in-the-middle
 * hooks are installed BEFORE the AWS SDK modules are loaded. This is critical
 * because the instrumentation patches @smithy/middleware-stack by returning a new
 * module object, which only works when the module hook intercepts the require().
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import { AwsInstrumentation } from "@opentelemetry/instrumentation-aws-sdk";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;
const OTLP_ENDPOINT = process.env.OTEL_EXPORTER_OTLP_ENDPOINT!;

async function main() {
  console.log("=== OTel Contrib JS: AWS Bedrock Conformance Test ===");

  // Set up OTel providers
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

  // Register instrumentation BEFORE importing AWS SDK modules.
  // registerInstrumentations installs require-in-the-middle hooks.
  registerInstrumentations({
    tracerProvider: provider,
    meterProvider: meterProvider,
    instrumentations: [new AwsInstrumentation()],
  });

  // Dynamic import AFTER registration so hooks can intercept the AWS SDK's
  // internal require('@smithy/middleware-stack') calls.
  const {
    BedrockRuntimeClient,
    ConverseCommand,
    ConverseStreamCommand,
    InvokeModelCommand,
  } = await import("@aws-sdk/client-bedrock-runtime");

  const client = new BedrockRuntimeClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
  });

  // Scenario: basic chat
  console.log("  [chat] basic chat completion");
  const resp = await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [{ role: "user", content: [{ text: "Say hello." }] }],
    })
  );
  console.log(`    -> ${resp.output?.message?.content?.[0]?.text?.slice(0, 60)}`);

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion");
  const streamResp = await client.send(
    new ConverseStreamCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [{ role: "user", content: [{ text: "Tell me a joke." }] }],
    })
  );
  let text = "";
  if (streamResp.stream) {
    for await (const event of streamResp.stream) {
      if (event.contentBlockDelta?.delta?.text) {
        text += event.contentBlockDelta.delta.text;
      }
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);

  // Scenario: embeddings
  console.log("  [embeddings] Bedrock Titan Embeddings");
  const embResp = await client.send(
    new InvokeModelCommand({
      modelId: "amazon.titan-embed-text-v2:0",
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify({ inputText: "Hello, world!" }),
    })
  );
  const embResult = JSON.parse(new TextDecoder().decode(embResp.body));
  console.log(`    -> embedding dim: ${embResult.embedding.length}`);

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
