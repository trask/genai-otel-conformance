/**
 * Conformance test: OpenLLMetry (Traceloop) JS LlamaIndex instrumentation.
 *
 * Exercises: chat
 * against a mock OpenAI server.
 */

import { LlamaIndexInstrumentation } from "@traceloop/instrumentation-llamaindex";
import type { OpenAI } from "@llamaindex/openai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";

async function runChat(llm: OpenAI) {
  console.log("  [chat] basic chat completion");
  const resp = await llm.chat({
    messages: [{ role: "user", content: "Say hello." }],
  });
  console.log(`    -> ${resp.message.content.toString().slice(0, 60)}`);
}

async function runEmbeddings(OpenAIEmbedding: any) {
  console.log("  [embeddings] embedding generation");
  const embed = new OpenAIEmbedding({
    model: "text-embedding-3-small",
    apiKey: "mock-key",
    additionalSessionOptions: { baseURL: MOCK_BASE_URL },
  });
  const result = await embed.getTextEmbedding("Hello, world!");
  console.log(`    -> embedding dim: ${result.length}`);
}

async function main() {
  console.log("=== OpenLLMetry JS: LlamaIndex Conformance Test ===");

  const otel = setupOtel();

  // Dynamic import AFTER OTel setup so manuallyInstrument can patch prototypes.
  // tsx (ESM) does not support require-in-the-middle, so enable() does not work.
  const llamaindexModule = await import("llamaindex");
  const openaiModule = await import("@llamaindex/openai");

  const instrumentation = new LlamaIndexInstrumentation();
  // manuallyInstrument patches the main llamaindex module (RetrieverQueryEngine, etc.)
  // Merge openaiModule so that OpenAI/OpenAIEmbedding classes are also patched
  // (they are not re-exported by the main llamaindex module).
  instrumentation.manuallyInstrument({ ...llamaindexModule, ...openaiModule } as any);

  const { Settings } = llamaindexModule;
  const { OpenAI, OpenAIEmbedding } = openaiModule;

  Settings.llm = new OpenAI({
    model: "gpt-4o-mini",
    additionalSessionOptions: { baseURL: MOCK_BASE_URL },
    apiKey: "mock-key",
  });

  const llm = new OpenAI({
    model: "gpt-4o-mini",
    additionalSessionOptions: { baseURL: MOCK_BASE_URL },
    apiKey: "mock-key",
  });

  await runChat(llm);
  await runEmbeddings(OpenAIEmbedding);

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); process.exit(1); });
