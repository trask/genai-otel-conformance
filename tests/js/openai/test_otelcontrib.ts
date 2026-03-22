/**
 * Conformance test: OTel Contrib (@opentelemetry/instrumentation-openai) JS OpenAI instrumentation.
 *
 * The @opentelemetry/instrumentation-openai package uses InstrumentationBase's
 * module hooks which don't work under tsx ESM. We work around this by:
 * 1. Setting up the tracer/meter providers on the instrumentation instance
 * 2. Calling init() to get module definitions and invoking patch() directly
 */

import { OpenAIInstrumentation } from "@opentelemetry/instrumentation-openai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL! + "/v1";

async function main() {
  console.log("=== OTel Contrib JS: OpenAI Conformance Test ===");

  // Set up OTel providers
  const otel = setupOtel();
  const { provider, meterProvider } = otel;

  // Create instrumentation and set providers so meter/tracer are available
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

  const OpenAI = openaiModule.default;
  const client = new OpenAI({ baseURL: MOCK_BASE_URL, apiKey: "mock-key" });

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

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
