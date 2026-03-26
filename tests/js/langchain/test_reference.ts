/**
 * Conformance test: Reference instrumentation for LangChain.
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
  console.log("=== Reference: LangChain JS Conformance Test ===");
  const otel = setupOtel();

  const { ChatOpenAI, OpenAIEmbeddings } = await import("@langchain/openai");

  const chatModel = "gpt-4o-mini";
  const llm = new ChatOpenAI({
    model: chatModel,
    apiKey: "mock-key",
    configuration: { baseURL: MOCK_BASE_URL },
  });

  // Scenario: chat
  console.log("  [chat] basic chat completion (reference)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    const resp = await llm.invoke([["user", "Say hello."]]);
    const responseMetadata = resp.response_metadata as any;
    if ((resp as any).id) {
      span.setAttribute("gen_ai.response.id", (resp as any).id);
    }
    if (responseMetadata?.model_name) {
      span.setAttribute("gen_ai.response.model", responseMetadata.model_name);
    }
    if (responseMetadata?.finish_reason) {
      span.setAttribute("gen_ai.response.finish_reasons", [responseMetadata.finish_reason]);
    }
    if (resp.usage_metadata) {
      span.setAttribute("gen_ai.usage.input_tokens", resp.usage_metadata.input_tokens);
      span.setAttribute("gen_ai.usage.output_tokens", resp.usage_metadata.output_tokens);
    }
    console.log(`    -> ${resp.content.toString().slice(0, 60)}`);
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion (reference)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    const stream = await llm.stream([["user", "Tell me a joke."]]);
    let text = "";
    let usageMetadata: any = null;
    let responseMetadata: any = null;
    let responseId: string | null = null;
    for await (const chunk of stream) {
      text += chunk.content;
      if (chunk.usage_metadata) usageMetadata = chunk.usage_metadata;
      if ((chunk as any).id) responseId = (chunk as any).id;
      if (chunk.response_metadata && Object.keys(chunk.response_metadata).length > 0) {
        responseMetadata = chunk.response_metadata;
      }
    }
    if (responseId) {
      span.setAttribute("gen_ai.response.id", responseId);
    }
    if (responseMetadata?.model_name) {
      span.setAttribute("gen_ai.response.model", responseMetadata.model_name);
    }
    if (responseMetadata?.finish_reason) {
      span.setAttribute("gen_ai.response.finish_reasons", [responseMetadata.finish_reason]);
    }
    if (usageMetadata) {
      span.setAttribute("gen_ai.usage.input_tokens", usageMetadata.input_tokens);
      span.setAttribute("gen_ai.usage.output_tokens", usageMetadata.output_tokens);
    }
    console.log(`    -> ${text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation (reference)");
  await tracer.startActiveSpan("embeddings text-embedding-3-small", async (span) => {
    const requestModel = "text-embedding-3-small";
    span.setAttribute("gen_ai.operation.name", "embeddings");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const embedModel = new OpenAIEmbeddings({
      model: requestModel,
      apiKey: "mock-key",
      configuration: { baseURL: MOCK_BASE_URL },
    });
    const result = await embedModel.embedQuery("Hello, world!");
    console.log(`    -> embedding dim: ${result.length}`);
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
