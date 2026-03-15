/**
 * Conformance test: OpenInference (Arize) JS Claude Agent SDK instrumentation.
 *
 * NOTE: There is no dedicated openinference-instrumentation-claude-agent-sdk
 * JS package. This test uses @arizeai/openinference-instrumentation-anthropic
 * which instruments the underlying Anthropic SDK used internally by the
 * Claude Agent SDK.
 *
 * The Claude Agent SDK in JS spawns a CLI subprocess, so a mock CLI script
 * (mock_cli.mjs) is used to simulate the protocol without requiring an API
 * key or the real Claude Code CLI.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import { AnthropicInstrumentation } from "@arizeai/openinference-instrumentation-anthropic";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

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

async function main() {
  console.log("=== OpenInference JS: Claude Agent SDK Conformance Test ===");

  const { provider, meterProvider, loggerProvider } = setupOtel();

  // Instrument the Anthropic SDK (used internally by the Claude Agent SDK)
  const anthropicModule = await import("@anthropic-ai/sdk");
  const instrumentation = new AnthropicInstrumentation();
  instrumentation.manuallyInstrument(anthropicModule);

  // Point to our mock CLI script
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const mockCliPath = join(__dirname, "mock_cli.mjs");

  const { query } = await import("@anthropic-ai/claude-agent-sdk");

  console.log("  [agent_query] basic query via mock CLI");
  const q = query({
    prompt: "Say hello.",
    options: {
      pathToClaudeCodeExecutable: mockCliPath,
      allowDangerouslySkipPermissions: true,
      permissionMode: "bypassPermissions" as any,
      maxTurns: 1,
      persistSession: false,
    },
  });

  for await (const message of q) {
    if (message.type === "assistant") {
      const content = (message as any).message?.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === "text") {
            console.log(`    -> ${block.text.slice(0, 60)}`);
          }
        }
      }
    } else if (message.type === "result") {
      console.log(`    -> result: turns=${(message as any).num_turns}`);
    }
  }

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
