/**
 * Conformance test: Prototype instrumentation for LangChain.
 */
import { z, toJSONSchema } from "zod";
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
  console.log("=== Prototype: LangChain JS Conformance Test ===");
  const otel = setupOtel();

  const { ChatOpenAI, OpenAIEmbeddings } = await import("@langchain/openai");

  const chatModel = "gpt-4o-mini";
  const llm = new ChatOpenAI({
    model: chatModel,
    apiKey: "mock-key",
    configuration: { baseURL: MOCK_BASE_URL },
  });

  // Scenario: chat
  console.log("  [chat] basic chat completion (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    const messages = [{ role: "user" as const, content: "Say hello." }];
    const resp = await llm.invoke(messages);
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

    // Emit inference operation details event
    logs.getLogger("gen_ai.prototype").emit({
      severityNumber: SeverityNumber.INFO,
      eventName: "gen_ai.client.inference.operation.details",
      body: "Inference operation details",
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": chatModel,
        "gen_ai.response.id": (resp as any).id,
        "gen_ai.response.model": responseMetadata?.model_name,
        "gen_ai.response.finish_reasons": responseMetadata?.finish_reason ? [responseMetadata.finish_reason] : undefined,
        "gen_ai.usage.input_tokens": resp.usage_metadata?.input_tokens,
        "gen_ai.usage.output_tokens": resp.usage_metadata?.output_tokens,
        "gen_ai.input.messages": JSON.stringify(
          messages.map(m => ({ role: m.role, parts: [{ type: "text", content: m.content }] }))
        ),
        "gen_ai.output.messages": JSON.stringify([{
          role: "assistant",
          parts: [{ type: "text", content: resp.content.toString() }],
          finish_reason: responseMetadata?.finish_reason,
        }]),
      },
    });

    console.log(`    -> ${resp.content.toString().slice(0, 60)}`);
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    const stream = await llm.stream([{ role: "user" as const, content: "Tell me a joke." }]);
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

  // Scenario: agent-style tool call request
  console.log("  [chat_tool_call] tool calling (prototype)");
  await tracer.startActiveSpan("chat gpt-4o-mini", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "openai");
    span.setAttribute("gen_ai.request.model", chatModel);
    const tools = [
      {
        name: "get_weather",
        description: "Get the current weather for a location.",
        schema: z.object({
          location: z.string().describe("The location to get weather for"),
        }),
      },
    ];
    // Derive from the same tools array passed to bindTools (LangChain StructuredToolParams format)
    span.setAttribute(
      "gen_ai.tool.definitions",
      JSON.stringify(tools.map((t) => ({ name: t.name, description: t.description, schema: toJSONSchema(t.schema) }))),
    );

    const llmWithTools = llm.bindTools(tools, { tool_choice: "auto" });
    const resp = await llmWithTools.invoke([{ role: "user" as const, content: "What's the weather in Seattle?" }]);
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
    if (Array.isArray((resp as any).tool_calls) && (resp as any).tool_calls.length > 0) {
      console.log(`    -> tool_call: ${(resp as any).tool_calls[0].name}`);
    } else {
      console.log(`    -> ${resp.content.toString().slice(0, 60)}`);
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
