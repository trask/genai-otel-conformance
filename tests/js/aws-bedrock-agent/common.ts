/**
 * Shared test infrastructure for AWS Bedrock Agent JS conformance tests.
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
import type { BedrockAgentRuntimeClient } from "@aws-sdk/client-bedrock-agent-runtime";

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

export async function runInvokeAgent(client: BedrockAgentRuntimeClient, InvokeAgentCommand: any) {
  console.log("  [invoke_agent] Bedrock Agent Runtime InvokeAgent");
  const resp = await client.send(
    new InvokeAgentCommand({
      agentId: "MOCK_AGENT_ID",
      agentAliasId: "MOCK_ALIAS_ID",
      sessionId: "mock-session-001",
      inputText: "Say hello.",
    })
  );

  let completionText = "";
  if (resp.completion) {
    for await (const event of resp.completion) {
      if (event.chunk?.bytes) {
        completionText += new TextDecoder().decode(event.chunk.bytes);
      }
    }
  }
  console.log(`    -> ${completionText.slice(0, 60)}`);
}

export async function run(title: string, instrumentFn: (agentModule: any) => void) {
  console.log(`=== ${title} ===`);

  const { provider, meterProvider, loggerProvider } = setupOtel();

  const agentModule = await import("@aws-sdk/client-bedrock-agent-runtime");
  instrumentFn(agentModule);

  const { BedrockAgentRuntimeClient, InvokeAgentCommand } = agentModule;
  const { NodeHttpHandler } = await import("@smithy/node-http-handler");
  const client = new BedrockAgentRuntimeClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
    requestHandler: new NodeHttpHandler(),
  });

  await runInvokeAgent(client, InvokeAgentCommand);

  console.log("Flushing telemetry...");
  await provider.forceFlush();
  await meterProvider.forceFlush();
  await loggerProvider.forceFlush();
  await provider.shutdown();
  await meterProvider.shutdown();
  await loggerProvider.shutdown();
  console.log("Done.");
}
