/**
 * Conformance test: Reference instrumentation for Cohere.
 */
import { trace } from "@opentelemetry/api";
import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL!;
const tracer = trace.getTracer("gen_ai.reference");

async function main() {
  console.log("=== Reference: Cohere JS Conformance Test ===");
  const otel = setupOtel();

  const { CohereClient } = await import("cohere-ai");
  const client = new CohereClient({ token: "mock-key", baseUrl: MOCK_BASE_URL });

  // Scenario: chat
  console.log("  [chat] basic chat completion (reference)");
  await tracer.startActiveSpan("chat command-r-plus", async (span) => {
    const requestModel = "command-r-plus";
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "cohere");
    span.setAttribute("gen_ai.request.model", requestModel);
    const resp = await client.chat({
      model: requestModel,
      message: "Say hello.",
    });
    if ((resp as any).generationId) {
      span.setAttribute("gen_ai.response.id", (resp as any).generationId);
    }
    if ((resp as any).finishReason) {
      span.setAttribute("gen_ai.response.finish_reasons", [(resp as any).finishReason]);
    }
    if ((resp as any).meta?.billedUnits?.inputTokens) {
      span.setAttribute("gen_ai.usage.input_tokens", (resp as any).meta.billedUnits.inputTokens);
    }
    if ((resp as any).meta?.billedUnits?.outputTokens) {
      span.setAttribute("gen_ai.usage.output_tokens", (resp as any).meta.billedUnits.outputTokens);
    }
    console.log(`    -> ${resp.text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation (reference)");
  await tracer.startActiveSpan("embeddings embed-english-v3.0", async (span) => {
    const requestModel = "embed-english-v3.0";
    span.setAttribute("gen_ai.operation.name", "embeddings");
    span.setAttribute("gen_ai.provider.name", "cohere");
    span.setAttribute("gen_ai.request.model", requestModel);
    const resp = await client.embed({
      model: requestModel,
      texts: ["Hello, world!"],
      inputType: "search_document",
    });
    if ((resp as any).meta?.billedUnits?.inputTokens) {
      span.setAttribute("gen_ai.usage.input_tokens", (resp as any).meta.billedUnits.inputTokens);
    }
    const embeddings = resp.embeddings as number[][];
    console.log(`    -> embedding dim: ${embeddings[0].length}`);
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
