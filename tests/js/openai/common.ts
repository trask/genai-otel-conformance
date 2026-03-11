/**
 * Shared test infrastructure for OpenAI JS conformance tests.
 */

import type OpenAI from "openai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";

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

  const otel = setupOtel();

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

  await flushAndShutdownOtel(otel);
}
