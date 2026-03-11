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

import { OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-vercel";
import { generateText, streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL! + "/v1";

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

  const otel = setupOtel({
    createSpanProcessors: (traceExporter) => [
      new OpenInferenceSimpleSpanProcessor({ exporter: traceExporter }),
    ],
  });

  const openai = createOpenAI({
    baseURL: MOCK_BASE_URL,
    apiKey: "mock-key",
  });

  await runChat(openai);
  await runChatStreaming(openai);

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
