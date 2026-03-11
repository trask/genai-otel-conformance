/**
 * Shared test infrastructure for Anthropic JS conformance tests.
 */

import type Anthropic from "@anthropic-ai/sdk";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;

export async function runChat(client: Anthropic) {
  console.log("  [chat] basic chat completion");
  const resp = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 100,
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.content[0].text.slice(0, 60)}`);
}

export async function runChatStreaming(client: Anthropic) {
  console.log("  [chat_streaming] streaming chat completion");
  const stream = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 100,
    messages: [{ role: "user", content: "Tell me a joke." }],
    stream: true,
  });
  let text = "";
  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      text += event.delta.text;
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);
}

export async function run(title: string, instrumentFn: (anthropicModule: any) => void) {
  console.log(`=== ${title} ===`);

  const otel = setupOtel();

  // Import @anthropic-ai/sdk and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const anthropicModule = await import("@anthropic-ai/sdk");
  instrumentFn(anthropicModule);

  const Anthropic = anthropicModule.default;
  const client = new Anthropic({ baseURL: MOCK_BASE_URL, apiKey: "mock-key" });

  await runChat(client);
  await runChatStreaming(client);

  await flushAndShutdownOtel(otel);
}
