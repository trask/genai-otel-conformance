/**
 * Shared test infrastructure for AWS Bedrock JS conformance tests.
 */

import type { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;

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
