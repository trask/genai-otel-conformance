/**
 * Conformance test: Prototype instrumentation for AWS Bedrock.
 */
import { trace } from "@opentelemetry/api";
import { logs, SeverityNumber } from "@opentelemetry/api-logs";
import { flushAndShutdownOtel, setupOtel } from "../otel";
import { runMemoryOperations } from "./common";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL!;
const tracer = trace.getTracer("gen_ai.prototype");

async function main() {
  console.log("=== Prototype: AWS Bedrock JS Conformance Test ===");
  const otel = setupOtel();

  const {
    BedrockRuntimeClient,
    ConverseCommand,
    ConverseStreamCommand,
    InvokeModelCommand,
  } = await import("@aws-sdk/client-bedrock-runtime");
  const { NodeHttpHandler } = await import("@smithy/node-http-handler");

  const client = new BedrockRuntimeClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
    requestHandler: new NodeHttpHandler(),
  });

  // Scenario: chat
  console.log("  [chat] basic chat completion (prototype)");
  await tracer.startActiveSpan("chat anthropic.claude-3-haiku-20240307-v1:0", async (span) => {
    const requestModel = "anthropic.claude-3-haiku-20240307-v1:0";
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "aws.bedrock");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("gen_ai.response.model", requestModel);
    const messages = [
      { role: "user" as const, content: [{ text: "Say hello." }] },
    ];
    const resp = await client.send(
      new ConverseCommand({
        modelId: requestModel,
        messages,
      })
    );
    if (resp.stopReason) {
      span.setAttribute("gen_ai.response.finish_reasons", [resp.stopReason]);
    }
    if (resp.usage) {
      if (resp.usage.inputTokens) span.setAttribute("gen_ai.usage.input_tokens", resp.usage.inputTokens);
      if (resp.usage.outputTokens) span.setAttribute("gen_ai.usage.output_tokens", resp.usage.outputTokens);
    }

    // Emit inference operation details event
    const outputText = resp.output?.message?.content?.[0]?.text ?? "";
    logs.getLogger("gen_ai.prototype").emit({
      severityNumber: SeverityNumber.INFO,
      eventName: "gen_ai.client.inference.operation.details",
      body: "Inference operation details",
      attributes: {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": requestModel,
        "gen_ai.response.model": requestModel,
        "gen_ai.response.finish_reasons": resp.stopReason ? [resp.stopReason] : undefined,
        "gen_ai.usage.input_tokens": resp.usage?.inputTokens,
        "gen_ai.usage.output_tokens": resp.usage?.outputTokens,
        "gen_ai.input.messages": JSON.stringify(
          messages.map(m => ({ role: m.role, parts: [{ type: "text", content: m.content[0].text }] }))
        ),
        "gen_ai.output.messages": JSON.stringify([{
          role: "assistant",
          parts: [{ type: "text", content: outputText }],
          finish_reason: resp.stopReason,
        }]),
      },
    });

    console.log(`    -> ${outputText.slice(0, 60)}`);
    span.end();
  });

  // Scenario: chat with tool calling
  console.log("  [chat_tool_call] chat with tool calling (prototype)");
  await tracer.startActiveSpan("chat anthropic.claude-3-haiku-20240307-v1:0", async (span) => {
    const requestModel = "anthropic.claude-3-haiku-20240307-v1:0";
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "aws.bedrock");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("gen_ai.response.model", requestModel);
    const toolSpec = {
      toolSpec: {
        name: "get_weather",
        description: "Get the current weather",
        inputSchema: {
          json: {
            type: "object",
            properties: {
              location: { type: "string", description: "City name" },
            },
            required: ["location"],
          },
        },
      },
    };
    const toolConfig = { tools: [toolSpec] };
    span.setAttribute("gen_ai.tool.definitions", JSON.stringify(toolConfig.tools));
    const messages = [
      { role: "user" as const, content: [{ text: "What's the weather in Seattle?" }] },
    ];
    const resp = await client.send(
      new ConverseCommand({
        modelId: requestModel,
        messages,
        toolConfig,
      })
    );
    if (resp.stopReason) {
      span.setAttribute("gen_ai.response.finish_reasons", [resp.stopReason]);
    }
    if (resp.usage) {
      if (resp.usage.inputTokens) span.setAttribute("gen_ai.usage.input_tokens", resp.usage.inputTokens);
      if (resp.usage.outputTokens) span.setAttribute("gen_ai.usage.output_tokens", resp.usage.outputTokens);
    }
    const content = resp.output?.message?.content?.[0];
    if (content && "toolUse" in content && content.toolUse) {
      console.log(`    -> tool_call: ${content.toolUse.name}`);
    } else {
      console.log(`    -> ${content?.text?.slice(0, 60) ?? ""}`);
    }
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion (prototype)");
  await tracer.startActiveSpan("chat anthropic.claude-3-haiku-20240307-v1:0", async (span) => {
    const requestModel = "anthropic.claude-3-haiku-20240307-v1:0";
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "aws.bedrock");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("gen_ai.response.model", requestModel);
    const resp = await client.send(
      new ConverseStreamCommand({
        modelId: requestModel,
        messages: [
          { role: "user", content: [{ text: "Tell me a joke." }] },
        ],
      })
    );
    let text = "";
    let stopReason = "";
    let inputTokens = 0;
    let outputTokens = 0;
    if (resp.stream) {
      for await (const event of resp.stream) {
        if (event.contentBlockDelta?.delta?.text) {
          text += event.contentBlockDelta.delta.text;
        }
        if (event.messageStop?.stopReason) {
          stopReason = event.messageStop.stopReason;
        }
        if (event.metadata?.usage) {
          inputTokens = event.metadata.usage.inputTokens ?? 0;
          outputTokens = event.metadata.usage.outputTokens ?? 0;
        }
      }
    }
    if (stopReason) span.setAttribute("gen_ai.response.finish_reasons", [stopReason]);
    if (inputTokens) span.setAttribute("gen_ai.usage.input_tokens", inputTokens);
    if (outputTokens) span.setAttribute("gen_ai.usage.output_tokens", outputTokens);
    console.log(`    -> ${text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: embeddings
  console.log("  [embeddings] Bedrock Titan Embeddings (prototype)");
  await tracer.startActiveSpan("embeddings amazon.titan-embed-text-v2:0", async (span) => {
    const requestModel = "amazon.titan-embed-text-v2:0";
    span.setAttribute("gen_ai.operation.name", "embeddings");
    span.setAttribute("gen_ai.provider.name", "aws.bedrock");
    span.setAttribute("gen_ai.request.model", requestModel);
    span.setAttribute("gen_ai.response.model", requestModel);
    const resp = await client.send(
      new InvokeModelCommand({
        modelId: requestModel,
        contentType: "application/json",
        accept: "application/json",
        body: JSON.stringify({ inputText: "Hello, world!" }),
      })
    );
    const result = JSON.parse(new TextDecoder().decode(resp.body));
    if (result.inputTextTokenCount) {
      span.setAttribute("gen_ai.usage.input_tokens", result.inputTextTokenCount);
    }
    console.log(`    -> embedding dim: ${result.embedding.length}`);
    span.end();
  });

  // Scenario: memory operations
  await runMemoryOperations();

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
