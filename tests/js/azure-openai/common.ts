/**
 * Shared test infrastructure for Azure OpenAI JS conformance tests.
 */

import type { AzureOpenAI } from "openai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;

export async function runChat(client: AzureOpenAI) {
  console.log("  [chat] basic chat completion");
  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.choices[0].message.content?.slice(0, 60)}`);
}

export async function runChatStreaming(client: AzureOpenAI) {
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

export async function runEmbeddings(client: AzureOpenAI) {
  console.log("  [embeddings] embedding generation");
  const resp = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: "Hello, world!",
  });
  console.log(`    -> embedding dim: ${resp.data[0].embedding.length}`);
}

export async function runChatToolCall(client: AzureOpenAI) {
  console.log("  [chat_tool_call] chat with tool calling");
  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "What's the weather in Seattle?" }],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get the current weather",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string", description: "City name" },
            },
            required: ["location"],
          },
        },
      },
    ],
  });
  const choice = resp.choices[0];
  if (choice.message.tool_calls?.length) {
    console.log(`    -> tool_call: ${choice.message.tool_calls[0].function.name}`);
  } else {
    console.log(`    -> ${choice.message.content?.slice(0, 60)}`);
  }
}

export async function run(title: string, instrumentFn: (openaiModule: any) => void) {
  console.log(`=== ${title} ===`);

  const otel = setupOtel();

  const openaiModule = await import("openai");
  instrumentFn(openaiModule);

  const client = new openaiModule.AzureOpenAI({
    endpoint: MOCK_BASE_URL,
    apiKey: "mock-key",
    apiVersion: "2024-06-01",
  });

  await runChat(client);
  await runChatStreaming(client);
  await runChatToolCall(client);
  await runEmbeddings(client);

  await flushAndShutdownOtel(otel);
}
