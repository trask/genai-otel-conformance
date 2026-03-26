/**
 * Conformance test: Reference instrumentation for Vercel AI SDK.
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
  console.log("=== Reference: Vercel AI SDK JS Conformance Test ===");
  const otel = setupOtel();

  const { generateText, streamText } = await import("ai");
  const { createOpenAI } = await import("@ai-sdk/openai");

  const openai = createOpenAI({
    baseURL: MOCK_BASE_URL,
    apiKey: "mock-key",
  });
  const requestModel = "gpt-4o-mini";

  // Scenario: chat
  console.log("  [chat] basic chat completion (reference)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const result = await generateText({
      model: openai.chat(requestModel),
      prompt: "Say hello.",
    });
    span.setAttribute("gen_ai.response.finish_reasons", [result.finishReason]);
    if (result.usage) {
      if (result.usage.inputTokens != null) {
        span.setAttribute("gen_ai.usage.input_tokens", result.usage.inputTokens);
      }
      if (result.usage.outputTokens != null) {
        span.setAttribute("gen_ai.usage.output_tokens", result.usage.outputTokens);
      }
    }
    if ((result as any).response?.model) {
      span.setAttribute("gen_ai.response.model", (result as any).response.model);
    } else {
      span.setAttribute("gen_ai.response.model", requestModel);
    }
    if ((result as any).response?.id) {
      span.setAttribute("gen_ai.response.id", (result as any).response.id);
    }
    console.log(`    -> ${result.text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion (reference)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const result = await streamText({
      model: openai.chat(requestModel),
      prompt: "Tell me a joke.",
    });
    let text = "";
    for await (const chunk of result.textStream) {
      text += chunk;
    }
    const usage = await result.usage;
    const finishReason = await result.finishReason;
    const response = await (result as any).response;
    span.setAttribute("gen_ai.response.finish_reasons", [finishReason]);
    if (usage) {
      if (usage.inputTokens != null) {
        span.setAttribute("gen_ai.usage.input_tokens", usage.inputTokens);
      }
      if (usage.outputTokens != null) {
        span.setAttribute("gen_ai.usage.output_tokens", usage.outputTokens);
      }
    }
    if (response?.model) {
      span.setAttribute("gen_ai.response.model", response.model);
    } else {
      span.setAttribute("gen_ai.response.model", requestModel);
    }
    if (response?.id) {
      span.setAttribute("gen_ai.response.id", response.id);
    }
    console.log(`    -> ${text.slice(0, 60)}`);
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
