/**
 * Conformance test: Prototype instrumentation for OpenAI.
 */
import { trace } from "@opentelemetry/api";
import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL! + "/v1";
const tracer = trace.getTracer("gen_ai.prototype");

async function main() {
  console.log("=== Prototype: OpenAI JS Conformance Test ===");
  const otel = setupOtel();

  const { default: OpenAI } = await import("openai");
  const client = new OpenAI({ baseURL: MOCK_BASE_URL, apiKey: "mock-key" });

  // Scenario: chat
  console.log("  [chat] basic chat completion (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    const requestModel = "gpt-4o-mini";
    const endpoint = new URL(MOCK_BASE_URL);
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("server.address", endpoint.hostname);
    if (endpoint.port) {
      span.setAttribute("server.port", Number(endpoint.port));
    }
    const resp = await client.chat.completions.create({
      model: requestModel,
      messages: [{ role: "user", content: "Say hello." }],
    });
    span.setAttribute("gen_ai.response.model", resp.model);
    span.setAttribute("gen_ai.response.id", resp.id);
    span.setAttribute("gen_ai.response.finish_reasons", resp.choices.map(c => c.finish_reason));
    if (resp.usage) {
      span.setAttribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens);
      span.setAttribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens);
    }
    console.log(`    -> ${resp.choices[0].message.content?.slice(0, 60)}`);
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    const requestModel = "gpt-4o-mini";
    const endpoint = new URL(MOCK_BASE_URL);
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("server.address", endpoint.hostname);
    if (endpoint.port) {
      span.setAttribute("server.port", Number(endpoint.port));
    }
    const stream = await client.chat.completions.create({
      model: requestModel,
      messages: [{ role: "user", content: "Tell me a joke." }],
      stream: true,
      stream_options: { include_usage: true },
    });
    let text = "";
    let model = "";
    let id = "";
    let finishReason = "";
    let inputTokens = 0;
    let outputTokens = 0;
    for await (const chunk of stream) {
      if (chunk.choices[0]?.delta?.content) {
        text += chunk.choices[0].delta.content;
      }
      if (chunk.model) model = chunk.model;
      if (chunk.id) id = chunk.id;
      if (chunk.choices[0]?.finish_reason) {
        finishReason = chunk.choices[0].finish_reason;
      }
      if (chunk.usage) {
        inputTokens = chunk.usage.prompt_tokens;
        outputTokens = chunk.usage.completion_tokens;
      }
    }
    span.setAttribute("gen_ai.response.model", model);
    span.setAttribute("gen_ai.response.id", id);
    if (finishReason) {
      span.setAttribute("gen_ai.response.finish_reasons", [finishReason]);
    }
    if (inputTokens) span.setAttribute("gen_ai.usage.input_tokens", inputTokens);
    if (outputTokens) span.setAttribute("gen_ai.usage.output_tokens", outputTokens);
    console.log(`    -> ${text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation (prototype)");
  await tracer.startActiveSpan("embeddings text-embedding-3-small", async (span) => {
    const requestModel = "text-embedding-3-small";
    const endpoint = new URL(MOCK_BASE_URL);
    span.setAttribute("gen_ai.operation.name", "embeddings");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("server.address", endpoint.hostname);
    if (endpoint.port) {
      span.setAttribute("server.port", Number(endpoint.port));
    }
    const resp = await client.embeddings.create({
      model: requestModel,
      input: "Hello, world!",
    });
    span.setAttribute("gen_ai.response.model", resp.model);
    if (resp.usage) {
      span.setAttribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens);
    }
    console.log(`    -> embedding dim: ${resp.data[0].embedding.length}`);
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
