/**
 * Conformance test: Vercel AI SDK with OTel telemetry.
 *
 * Exercises: chat, chat_streaming
 * against a mock OpenAI server, with Vercel AI SDK's built-in OTel support.
 *
 * Vercel AI SDK supports `experimental_telemetry` to emit OTel spans.
 */

import { generateText, streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";

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
  console.log("=== Vercel AI SDK: Conformance Test ===");

  const otel = setupOtel();

  const openai = createOpenAI({
    baseURL: MOCK_BASE_URL,
    apiKey: "mock-key",
  });

  await runChat(openai);
  await runChatStreaming(openai);

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); process.exit(1); });
