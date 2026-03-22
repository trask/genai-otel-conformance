/**
 * Shared test infrastructure for Cohere JS conformance tests.
 */

import type { CohereClient } from "cohere-ai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;

export async function runChat(client: any) {
  console.log("  [chat] basic chat completion");
  const resp = await client.chat({
    model: "command-r-plus",
    message: "Say hello.",
  });
  console.log(`    -> ${resp.text.slice(0, 60)}`);
}

export async function runEmbeddings(client: any) {
  console.log("  [embeddings] embedding generation");
  const resp = await client.embed({
    model: "embed-english-v3.0",
    texts: ["Hello, world!"],
    inputType: "search_document",
  });
  const embeddings = resp.embeddings as number[][];
  console.log(`    -> embedding dim: ${embeddings[0].length}`);
}

export async function run(title: string, instrumentFn: (cohereModule: any) => void) {
  console.log(`=== ${title} ===`);

  const otel = setupOtel();

  // Import cohere-ai and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const cohereModule = await import("cohere-ai");
  instrumentFn(cohereModule);

  const { CohereClient } = cohereModule;
  const client = new CohereClient({ token: "mock-key", baseUrl: MOCK_BASE_URL });

  await runChat(client);
  await runEmbeddings(client);

  await flushAndShutdownOtel(otel);
}
