/**
 * Conformance test: Prototype instrumentation for Vercel AI SDK.
 */
import { trace } from "@opentelemetry/api";
import { logs, SeverityNumber } from "@opentelemetry/api-logs";
import { tool } from "ai";
import { z, toJSONSchema } from "zod";
import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL! + "/v1";
const tracer = trace.getTracer("gen_ai.prototype");

async function main() {
  console.log("=== Prototype: Vercel AI SDK JS Conformance Test ===");
  const otel = setupOtel();

  const { generateText, streamText } = await import("ai");
  const { createOpenAI } = await import("@ai-sdk/openai");

  const openai = createOpenAI({
    baseURL: MOCK_BASE_URL,
    apiKey: "mock-key",
  });
  const requestModel = "gpt-4o-mini";

  // Scenario: chat
  console.log("  [chat] basic chat completion (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const userMessage = "Say hello.";
    const result = await generateText({
      model: openai.chat(requestModel),
      prompt: userMessage,
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

    // Emit inference operation details event
    logs.getLogger("gen_ai.prototype").emit({
      severityNumber: SeverityNumber.INFO,
      eventName: "gen_ai.client.inference.operation.details",
      body: "Inference operation details",
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": requestModel,
        "gen_ai.response.id": (result as any).response?.id,
        "gen_ai.response.model": (result as any).response?.model ?? requestModel,
        "gen_ai.response.finish_reasons": [result.finishReason],
        "gen_ai.usage.input_tokens": result.usage?.inputTokens,
        "gen_ai.usage.output_tokens": result.usage?.outputTokens,
        "gen_ai.input.messages": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: userMessage }] }
        ]),
        "gen_ai.output.messages": JSON.stringify([{
          role: "assistant",
          parts: [{ type: "text", content: result.text }],
          finish_reason: result.finishReason,
        }]),
      },
    });

    console.log(`    -> ${result.text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion (prototype)");
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

  // Scenario: chat with tool calling
  console.log("  [chat_tool_call] chat with tool calling (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const weatherToolSchema = z.object({
      location: z.string().describe("The location to get weather for"),
    });
    const tools = {
      get_weather: tool({
        description: "Get the current weather for a location.",
        inputSchema: weatherToolSchema,
        execute: async ({ location }) => ({
          location: location,
          weather: "Sunny, 72°F",
        }),
      }),
    };
    span.setAttribute(
      "gen_ai.tool.definitions",
      JSON.stringify(
        Object.entries(tools).map(([name, t]) => ({
          name: name,
          description: t.description,
          inputSchema: toJSONSchema(t.inputSchema!),
        })),
      ),
    );
    const result = await generateText({
      model: openai.chat(requestModel),
      prompt: "What's the weather in Seattle?",
      tools,
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
    const toolCalls = (result as any).toolCalls;
    if (Array.isArray(toolCalls) && toolCalls.length > 0) {
      console.log(`    -> tool_call: ${toolCalls[0].toolName}`);
    } else {
      console.log(`    -> ${result.text.slice(0, 60)}`);
    }
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
