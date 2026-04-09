/**
 * Conformance test: Prototype instrumentation for LlamaIndex.
 */
import { trace } from "@opentelemetry/api";
import { logs, SeverityNumber } from "@opentelemetry/api-logs";
import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL! + "/v1";
const tracer = trace.getTracer("gen_ai.prototype");

async function main() {
  console.log("=== Prototype: LlamaIndex JS Conformance Test ===");
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
  console.log("  [chat] basic chat completion (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    span.setAttribute("gen_ai.request.top_p", requestTopP);
    const userMessage = "Say hello.";
    const resp = await llm.chat({
      messages: [{ role: "user", content: userMessage }],
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

    // Emit inference operation details event
    logs.getLogger("gen_ai.prototype").emit({
      severityNumber: SeverityNumber.INFO,
      eventName: "gen_ai.client.inference.operation.details",
      body: "Inference operation details",
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": chatModel,
        "gen_ai.response.id": raw?.id,
        "gen_ai.response.model": raw?.model,
        "gen_ai.response.finish_reasons": raw?.choices?.[0]?.finish_reason ? [raw.choices[0].finish_reason] : undefined,
        "gen_ai.usage.input_tokens": raw?.usage?.prompt_tokens,
        "gen_ai.usage.output_tokens": raw?.usage?.completion_tokens,
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: userMessage }] }
        ]),
        "gen_ai.output.messages": JSON.stringify([{
          role: "assistant",
          parts: [{ type: "text", content: resp.message.content.toString() }],
          finish_reason: raw?.choices?.[0]?.finish_reason,
        }]),
      },
    });

    console.log(`    -> ${resp.message.content.toString().slice(0, 60)}`);
    span.end();
  });

  // Scenario: chat with tool calling
  console.log("  [chat_tool_call] chat with tool calling (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    const requestTool = {
      type: "function" as const,
      function: {
        name: "get_weather",
        description: "Get the current weather",
        parameters: {
          type: "object",
          properties: {
            location: { type: "string", description: "City name" },
          },
          required: ["location"],
        },
      },
    };
    span.setAttribute("gen_ai.tool.definitions", JSON.stringify([requestTool]));
    const resp = await llm.chat({
      messages: [{ role: "user", content: "What's the weather in Seattle?" }],
      additionalChatOptions: { tools: [requestTool] },
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
    if (raw?.choices?.[0]?.message?.tool_calls?.length) {
      console.log(`    -> tool_call: ${raw.choices[0].message.tool_calls[0].function.name}`);
    } else {
      console.log(`    -> ${resp.message.content.toString().slice(0, 60)}`);
    }
    span.end();
  });

  // Scenario: embeddings
  console.log("  [embeddings] embedding generation (prototype)");
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
