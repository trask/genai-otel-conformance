/**
 * Conformance test: Prototype instrumentation for Cohere.
 */
import { trace } from "@opentelemetry/api";
import { logs, SeverityNumber } from "@opentelemetry/api-logs";
import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL!;
const tracer = trace.getTracer("gen_ai.prototype");

async function main() {
  console.log("=== Prototype: Cohere JS Conformance Test ===");
  const otel = setupOtel();

  const { CohereClient } = await import("cohere-ai");
  const client = new CohereClient({ token: "mock-key", baseUrl: MOCK_BASE_URL });

  // Scenario: chat
  console.log("  [chat] basic chat completion (prototype)");
  await tracer.startActiveSpan("chat command-r-plus", async (span) => {
    const requestModel = "command-r-plus";
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "cohere");
    span.setAttribute("gen_ai.request.model", requestModel);
    const userMessage = "Say hello.";
    const resp = await client.chat({
      model: requestModel,
      message: userMessage,
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

    // Emit inference operation details event
    logs.getLogger("gen_ai.prototype").emit({
      severityNumber: SeverityNumber.INFO,
      eventName: "gen_ai.client.inference.operation.details",
      body: "Inference operation details",
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": requestModel,
        "gen_ai.response.id": (resp as any).generationId,
        "gen_ai.response.finish_reasons": (resp as any).finishReason ? [(resp as any).finishReason] : undefined,
        "gen_ai.usage.input_tokens": (resp as any).meta?.billedUnits?.inputTokens,
        "gen_ai.usage.output_tokens": (resp as any).meta?.billedUnits?.outputTokens,
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: userMessage }] }
        ]),
        "gen_ai.output.messages": JSON.stringify([{
          role: "assistant",
          parts: [{ type: "text", content: resp.text }],
          finish_reason: (resp as any).finishReason,
        }]),
      },
    });

    console.log(`    -> ${resp.text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: chat with tool calling
  console.log("  [chat_tool_call] chat with tool calling (prototype)");
  await tracer.startActiveSpan("chat command-r-plus", async (span) => {
    const requestModel = "command-r-plus";
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "cohere");
    span.setAttribute("gen_ai.request.model", requestModel);
    const requestTool = {
      name: "get_weather",
      description: "Get the current weather",
      parameterDefinitions: {
        location: { description: "City name", type: "str" as const, required: true },
      },
    };
    span.setAttribute("gen_ai.tool.definitions", JSON.stringify([requestTool]));
    const resp = await client.chat({
      model: requestModel,
      message: "What's the weather in Seattle?",
      tools: [requestTool],
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
    if ((resp as any).toolCalls?.length) {
      console.log(`    -> tool_call: ${(resp as any).toolCalls[0].name}`);
    } else {
      console.log(`    -> ${resp.text.slice(0, 60)}`);
    }
    span.end();
  });

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation (prototype)");
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
