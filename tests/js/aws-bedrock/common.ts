/**
 * Shared test infrastructure for AWS Bedrock JS conformance tests.
 */

import { trace, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import type { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";

import { flushAndShutdownOtel, setupOtel } from "../otel";

export { setupOtel };

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;
const _parsedUrl = new URL(MOCK_BASE_URL);
const _SERVER_ADDRESS = _parsedUrl.hostname;
const _SERVER_PORT = parseInt(_parsedUrl.port) || 443;

const memoryTracer = trace.getTracer("gen_ai.memory.aws_bedrock");

export async function runChat(client: BedrockRuntimeClient, ConverseCommand: any) {
  console.log("  [chat] basic chat completion");
  const resp = await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [
        {
          role: "user",
          content: [{ text: "Say hello." }],
        },
      ],
    })
  );
  console.log(`    -> ${resp.output?.message?.content?.[0]?.text?.slice(0, 60)}`);
}

export async function runChatStreaming(client: BedrockRuntimeClient, ConverseStreamCommand: any) {
  console.log("  [chat_streaming] streaming chat completion");
  const resp = await client.send(
    new ConverseStreamCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [
        {
          role: "user",
          content: [{ text: "Tell me a joke." }],
        },
      ],
    })
  );
  let text = "";
  if (resp.stream) {
    for await (const event of resp.stream) {
      if (event.contentBlockDelta?.delta?.text) {
        text += event.contentBlockDelta.delta.text;
      }
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);
}

export async function runEmbeddings(client: BedrockRuntimeClient, InvokeModelCommand: any) {
  console.log("  [embeddings] Bedrock Titan Embeddings");
  const resp = await client.send(
    new InvokeModelCommand({
      modelId: "amazon.titan-embed-text-v2:0",
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify({ inputText: "Hello, world!" }),
    })
  );
  const result = JSON.parse(new TextDecoder().decode(resp.body));
  console.log(`    -> embedding dim: ${result.embedding.length}`);
}

export async function runChatToolCall(client: BedrockRuntimeClient, ConverseCommand: any) {
  console.log("  [chat_tool_call] Bedrock Converse with tool calling");
  const resp = await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [
        {
          role: "user",
          content: [{ text: "What's the weather in Seattle?" }],
        },
      ],
      toolConfig: {
        tools: [
          {
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
          },
        ],
      },
    })
  );
  const content = resp.output?.message?.content;
  if (content?.[0]?.toolUse) {
    console.log(`    -> tool_call: ${content[0].toolUse.name}`);
  } else {
    console.log(`    -> ${content?.[0]?.text?.slice(0, 60)}`);
  }
}

/**
 * Exercise Bedrock AgentCore Memory operations for conformance testing.
 *
 * Uses the actual @aws-sdk/client-bedrock-agentcore SDK with prototype OTel
 * spans to demonstrate which gen_ai.memory.* attributes are capturable.
 */
export async function runMemoryOperations() {
  const {
    BedrockAgentCoreClient,
    BatchCreateMemoryRecordsCommand,
    RetrieveMemoryRecordsCommand,
    BatchDeleteMemoryRecordsCommand,
  } = await import("@aws-sdk/client-bedrock-agentcore");
  const {
    BedrockAgentCoreControlClient,
    CreateMemoryCommand,
    DeleteMemoryCommand,
  } = await import("@aws-sdk/client-bedrock-agentcore-control");
  const { NodeHttpHandler } = await import("@smithy/node-http-handler");

  const agentcore = new BedrockAgentCoreClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
    requestHandler: new NodeHttpHandler(),
  });

  const controlClient = new BedrockAgentCoreControlClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
    requestHandler: new NodeHttpHandler(),
  });

  const memoryName = "conformance-test-memory-store";

  function setCommonAttrs(span: any, operationName: string, memoryId: string) {
    span.setAttribute("gen_ai.operation.name", operationName);
    span.setAttribute("gen_ai.provider.name", "aws.bedrock");
    span.setAttribute("gen_ai.memory.store.id", memoryId);
    span.setAttribute("server.address", _SERVER_ADDRESS);
    span.setAttribute("server.port", _SERVER_PORT);
  }

  // 0. Create memory store (create_memory_store span)
  console.log("  [create_memory_store] Bedrock AgentCore CreateMemory");
  let memoryId = "";
  await memoryTracer.startActiveSpan("create_memory_store", { kind: SpanKind.CLIENT }, async (span) => {
    span.setAttribute("gen_ai.operation.name", "create_memory_store");
    span.setAttribute("gen_ai.provider.name", "aws.bedrock");
    span.setAttribute("server.address", _SERVER_ADDRESS);
    span.setAttribute("server.port", _SERVER_PORT);
    const eventExpiryDuration = 86400;
    const expiration = new Date(Date.now() + eventExpiryDuration * 1000).toISOString();
    span.setAttribute("gen_ai.memory.expiration_date", expiration);
    try {
      const createMemoryResp = await controlClient.send(new CreateMemoryCommand({ name: memoryName, eventExpiryDuration }));
      memoryId = createMemoryResp.memory?.id ?? "";
      span.setAttribute("gen_ai.memory.store.id", memoryId);
      console.log(`    -> created memory store: ${memoryId}`);
    } catch (e: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: e.message });
      span.setAttribute("error.type", e.constructor?.name ?? "Error");
      throw e;
    } finally {
      span.end();
    }
  });

  // 1. Create memory records (update_memory span)
  console.log("  [update_memory] Bedrock AgentCore BatchCreateMemoryRecords");
  const now = new Date();
  const recordsInput = [
    {
      requestIdentifier: "req-001",
      namespaces: ["conformance-test"],
      content: { text: "The user prefers concise answers." },
      timestamp: now,
    },
    {
      requestIdentifier: "req-002",
      namespaces: ["conformance-test"],
      content: { text: "The user's name is Alice." },
      timestamp: now,
    },
  ];

  let recordIds: string[] = [];
  await memoryTracer.startActiveSpan("update_memory", { kind: SpanKind.CLIENT }, async (span) => {
    setCommonAttrs(span, "update_memory", memoryId);
    span.setAttribute("gen_ai.memory.record.content",
      recordsInput.map(r => r.content.text).join("; "));
    try {
      const createResp = await agentcore.send(new BatchCreateMemoryRecordsCommand({
        memoryId,
        records: recordsInput,
      }));
      recordIds = (createResp.successfulRecords ?? []).map((r: any) => r.memoryRecordId);
      if (recordIds.length > 0) {
        span.setAttribute("gen_ai.memory.record.id", recordIds[0]);
      }
      console.log(`    -> created ${recordIds.length} records: ${JSON.stringify(recordIds)}`);
    } catch (e: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: e.message });
      span.setAttribute("error.type", e.constructor?.name ?? "Error");
      throw e;
    } finally {
      span.end();
    }
  });

  // 2. Retrieve memory records (search_memory span)
  console.log("  [search_memory] Bedrock AgentCore RetrieveMemoryRecords");
  const searchQuery = "What does the user prefer?";
  await memoryTracer.startActiveSpan("search_memory", { kind: SpanKind.CLIENT }, async (span) => {
    setCommonAttrs(span, "search_memory", memoryId);
    span.setAttribute("gen_ai.memory.query.text", searchQuery);
    try {
      const retrieveResp = await agentcore.send(new RetrieveMemoryRecordsCommand({
        memoryId,
        namespace: "conformance-test",
        searchCriteria: { searchQuery },
      }));
      const summaries = retrieveResp.memoryRecordSummaries ?? [];
      span.setAttribute("gen_ai.memory.search.result.count", summaries.length);
      console.log(`    -> retrieved ${summaries.length} records`);
    } catch (e: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: e.message });
      span.setAttribute("error.type", e.constructor?.name ?? "Error");
      throw e;
    } finally {
      span.end();
    }
  });

  // 3. Delete memory records (delete_memory span)
  console.log("  [delete_memory] Bedrock AgentCore BatchDeleteMemoryRecords");
  await memoryTracer.startActiveSpan("delete_memory", { kind: SpanKind.CLIENT }, async (span) => {
    setCommonAttrs(span, "delete_memory", memoryId);
    if (recordIds.length > 0) {
      span.setAttribute("gen_ai.memory.record.id", recordIds[0]);
    }
    try {
      const deleteResp = await agentcore.send(new BatchDeleteMemoryRecordsCommand({
        memoryId,
        records: recordIds.slice(0, 1).map((id: string) => ({ memoryRecordId: id })),
      }));
      console.log(`    -> deleted ${(deleteResp.successfulRecords ?? []).length} records`);
    } catch (e: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: e.message });
      span.setAttribute("error.type", e.constructor?.name ?? "Error");
      throw e;
    } finally {
      span.end();
    }
  });

  // 4. Delete memory store (delete_memory_store span)
  console.log("  [delete_memory_store] Bedrock AgentCore DeleteMemory");
  await memoryTracer.startActiveSpan("delete_memory_store", { kind: SpanKind.CLIENT }, async (span) => {
    setCommonAttrs(span, "delete_memory_store", memoryId);
    try {
      await controlClient.send(new DeleteMemoryCommand({ memoryId }));
      console.log(`    -> deleted memory store: ${memoryId}`);
    } catch (e: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: e.message });
      span.setAttribute("error.type", e.constructor?.name ?? "Error");
      throw e;
    } finally {
      span.end();
    }
  });

  agentcore.destroy();
  controlClient.destroy();
}

export async function run(title: string, instrumentFn: (bedrockModule: any) => void) {
  console.log(`=== ${title} ===`);

  const otel = setupOtel();

  // Import AWS Bedrock SDK and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const bedrockModule = await import("@aws-sdk/client-bedrock-runtime");
  instrumentFn(bedrockModule);

  const { BedrockRuntimeClient, ConverseCommand, ConverseStreamCommand, InvokeModelCommand } = bedrockModule;
  const { NodeHttpHandler } = await import("@smithy/node-http-handler");
  const client = new BedrockRuntimeClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
    requestHandler: new NodeHttpHandler(),
  });

  await runChat(client, ConverseCommand);
  await runChatStreaming(client, ConverseStreamCommand);
  await runChatToolCall(client, ConverseCommand);
  await runEmbeddings(client, InvokeModelCommand);

  await flushAndShutdownOtel(otel);
}
