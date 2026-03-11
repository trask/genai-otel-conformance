/**
 * Conformance test: OpenInference (Arize) JS LangChain instrumentation.
 *
 * Exercises: chat, chat_streaming
 * against a mock OpenAI server.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-grpc";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-grpc";
import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";

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

async function main() {
  console.log("=== OpenInference JS: LangChain Conformance Test ===");

  const { provider, meterProvider, loggerProvider } = setupOtel();

  // Dynamic import AFTER OTel setup so manuallyInstrument can patch prototypes.
  // tsx (ESM) does not support require-in-the-middle, so enable() does not work.
  // The OpenInference LangChain instrumentation patches CallbackManager from
  // @langchain/core/dist/callbacks/manager.cjs, so we must import that module.
  const callbacksModule = await import("@langchain/core/callbacks/manager");
  const { ChatOpenAI } = await import("@langchain/openai");
  const { StringOutputParser } = await import("@langchain/core/output_parsers");
  const { ChatPromptTemplate } = await import("@langchain/core/prompts");

  const instrumentation = new LangChainInstrumentation();
  instrumentation.manuallyInstrument(callbacksModule);

  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    apiKey: "mock-key",
    configuration: { baseURL: MOCK_BASE_URL },
  });

  // Use RunnableSequence (pipe) so LangChain instrumentation can capture spans
  const prompt = ChatPromptTemplate.fromMessages([["user", "{input}"]]);
  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  console.log("  [chat] basic chat completion via chain");
  const resp = await chain.invoke({ input: "Say hello." });
  console.log(`    -> ${resp.slice(0, 60)}`);

  console.log("  [chat_streaming] streaming chat completion via chain");
  let text = "";
  const stream = await chain.stream({ input: "Tell me a joke." });
  for await (const chunk of stream) {
    text += chunk;
  }
  console.log(`    -> ${text.slice(0, 60)}`);

  console.log("  [embeddings] embedding generation");
  const { OpenAIEmbeddings } = await import("@langchain/openai");
  const embedModel = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: "mock-key",
    configuration: { baseURL: MOCK_BASE_URL },
  });
  const embResult = await embedModel.embedQuery("Hello, world!");
  console.log(`    -> embedding dim: ${embResult.length}`);

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
