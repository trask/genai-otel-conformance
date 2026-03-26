/**
 * Conformance test: Reference instrumentation for Anthropic.
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
  console.log("=== Reference: Anthropic JS Conformance Test ===");
  const otel = setupOtel();

  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic({ baseURL: MOCK_BASE_URL, apiKey: "mock-key" });

  // Scenario: chat
  console.log("  [chat] basic chat completion (reference)");
  await tracer.startActiveSpan("chat claude-sonnet-4-20250514", async (span) => {
    const requestModel = "claude-sonnet-4-20250514";
    const requestMaxTokens = 100;
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "anthropic");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("gen_ai.request.max_tokens", requestMaxTokens);
    const resp = await client.messages.create({
      model: requestModel,
      max_tokens: requestMaxTokens,
      messages: [{ role: "user", content: "Say hello." }],
    });
    span.setAttribute("gen_ai.response.model", resp.model);
    span.setAttribute("gen_ai.response.id", resp.id);
    span.setAttribute("gen_ai.response.finish_reasons", [resp.stop_reason ?? "end_turn"]);
    if (resp.usage) {
      span.setAttribute("gen_ai.usage.input_tokens", resp.usage.input_tokens);
      span.setAttribute("gen_ai.usage.output_tokens", resp.usage.output_tokens);
    }
    console.log(`    -> ${(resp.content[0] as any).text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion (reference)");
  await tracer.startActiveSpan("chat claude-sonnet-4-20250514", async (span) => {
    const requestModel = "claude-sonnet-4-20250514";
    const requestMaxTokens = 100;
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "anthropic");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("gen_ai.request.max_tokens", requestMaxTokens);
    const stream = await client.messages.create({
      model: requestModel,
      max_tokens: requestMaxTokens,
      messages: [{ role: "user", content: "Tell me a joke." }],
      stream: true,
    });
    let text = "";
    let model = "";
    let id = "";
    let stopReason = "";
    let inputTokens = 0;
    let outputTokens = 0;
    for await (const event of stream) {
      if (event.type === "message_start") {
        model = (event as any).message.model;
        id = (event as any).message.id;
        if ((event as any).message.usage) {
          inputTokens = (event as any).message.usage.input_tokens;
        }
      } else if (
        event.type === "content_block_delta" &&
        (event as any).delta.type === "text_delta"
      ) {
        text += (event as any).delta.text;
      } else if (event.type === "message_delta") {
        if ((event as any).delta.stop_reason) {
          stopReason = (event as any).delta.stop_reason;
        }
        if ((event as any).usage) {
          outputTokens = (event as any).usage.output_tokens;
        }
      }
    }
    if (model) span.setAttribute("gen_ai.response.model", model);
    if (id) span.setAttribute("gen_ai.response.id", id);
    if (stopReason) span.setAttribute("gen_ai.response.finish_reasons", [stopReason]);
    if (inputTokens) span.setAttribute("gen_ai.usage.input_tokens", inputTokens);
    if (outputTokens) span.setAttribute("gen_ai.usage.output_tokens", outputTokens);
    console.log(`    -> ${text.slice(0, 60)}`);
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
