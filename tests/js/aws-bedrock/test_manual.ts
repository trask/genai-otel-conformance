/**
 * Conformance test: Manual memory instrumentation for AWS Bedrock.
 *
 * Exercises memory operations (create_memory_store, update_memory,
 * search_memory, delete_memory, delete_memory_store) with manual OTel spans.
 */

import { setupOtel, runMemoryOperations } from "./common";

async function main() {
  console.log("=== Manual: AWS Bedrock JS Memory Conformance Test ===");

  const { provider, meterProvider, loggerProvider } = setupOtel();

  await runMemoryOperations();

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
