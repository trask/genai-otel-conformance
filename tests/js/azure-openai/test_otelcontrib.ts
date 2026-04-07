/**
 * Conformance test: OTel Contrib (@opentelemetry/instrumentation-openai) JS OpenAI instrumentation
 * with Azure OpenAI.
 *
 * Uses the same InstrumentationBase manual patching approach as the OpenAI otelcontrib test.
 */

import { OpenAIInstrumentation } from "@opentelemetry/instrumentation-openai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;

async function main() {
  console.log("=== OTel Contrib JS: Azure OpenAI Conformance Test ===");

  // Set up OTel providers
  const otel = setupOtel();
  const { provider, meterProvider } = otel;

  // Create instrumentation and set providers
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.setTracerProvider(provider);
  instrumentation.setMeterProvider(meterProvider);

  // Import openai then manually patch
  const openaiModule = await import("openai");
  const defs = (instrumentation as any).init();
  for (const def of defs) {
    if (typeof def.patch === "function") {
      def.patch(openaiModule);
    }
  }

  const client = new openaiModule.AzureOpenAI({
    endpoint: MOCK_BASE_URL,
    apiKey: "mock-key",
    apiVersion: "2024-06-01",
  });

  // Scenario: basic chat
  console.log("  [chat] basic chat completion");
  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.choices[0].message.content?.slice(0, 60)}`);

  // Scenario: streaming chat
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

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation");
  const embResp = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: "Hello, world!",
  });
  console.log(`    -> embedding dim: ${embResp.data[0].embedding.length}`);

  // Scenario: chat with tool calling
  console.log("  [chat_tool_call] chat with tool calling");
  const toolResp = await client.chat.completions.create({
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
  const toolChoice = toolResp.choices[0];
  if (toolChoice.message.tool_calls?.length) {
    console.log(`    -> tool_call: ${toolChoice.message.tool_calls[0].function.name}`);
  } else {
    console.log(`    -> ${toolChoice.message.content?.slice(0, 60)}`);
  }

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); process.exit(1); });
