/**
 * Conformance test: Reference instrumentation for LlamaIndex.
 */
import { trace } from "@opentelemetry/api";
import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL! + "/v1";
const tracer = trace.getTracer("gen_ai.reference");

async function main() {
  console.log("=== Reference: LlamaIndex JS Conformance Test ===");
  const otel = setupOtel();

  const { OpenAI, OpenAIEmbedding } = await import("@llamaindex/openai");

  const chatModel = "gpt-4o-mini";
  const requestTopP = 1.0;
  const llm = new OpenAI({
    model: chatModel,
    topP: requestTopP,
    additionalSessionOptions: { baseURL: MOCK_BASE_URL },
    apiKey: "mock-key",
  });

  // Scenario: chat
  console.log("  [chat] basic chat completion (reference)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    span.setAttribute("gen_ai.request.top_p", requestTopP);
    const resp = await llm.chat({
      messages: [{ role: "user", content: "Say hello." }],
    });
    const raw = resp.raw as any;
    if (raw?.model) span.setAttribute("gen_ai.response.model", raw.model);
    if (raw?.id) span.setAttribute("gen_ai.response.id", raw.id);
    if (raw?.choices?.[0]?.finish_reason) {
      span.setAttribute("gen_ai.response.finish_reasons", [raw.choices[0].finish_reason]);
    }
    if (raw?.usage) {
      span.setAttribute("gen_ai.usage.input_tokens", raw.usage.prompt_tokens);
      span.setAttribute("gen_ai.usage.output_tokens", raw.usage.completion_tokens);
    }
    console.log(`    -> ${resp.message.content.toString().slice(0, 60)}`);
    span.end();
  });

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation (reference)");
  await tracer.startActiveSpan("embeddings text-embedding-3-small", async (span) => {
    const requestModel = "text-embedding-3-small";
    span.setAttribute("gen_ai.operation.name", "embeddings");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const embed = new OpenAIEmbedding({
      model: requestModel,
      apiKey: "mock-key",
      additionalSessionOptions: { baseURL: MOCK_BASE_URL },
    });
    const result = await embed.getTextEmbedding("Hello, world!");
    console.log(`    -> embedding dim: ${result.length}`);
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
