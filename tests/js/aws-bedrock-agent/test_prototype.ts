/**
 * Conformance test: Manual invoke_agent instrumentation for AWS Bedrock Agent.
 *
 * Exercises: invoke_agent (Bedrock Agent Runtime InvokeAgent API)
 * with manual OTel spans.
 */

import { setupOtel, runInvokeAgent } from "./common";
import { flushAndShutdownOtel } from "../otel";

async function main() {
  console.log("=== Manual: AWS Bedrock Agent JS Invoke Agent Conformance Test ===");

  const otel = setupOtel();

  await runInvokeAgent();

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); process.exit(1); });
