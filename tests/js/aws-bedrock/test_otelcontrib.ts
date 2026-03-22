/**
 * Conformance test: OTel Contrib (@opentelemetry/instrumentation-aws-sdk) JS AWS Bedrock instrumentation.
 *
 * Uses registerInstrumentations + dynamic import() so that require-in-the-middle
 * hooks are installed BEFORE the AWS SDK modules are loaded. This is critical
 * because the instrumentation patches @smithy/middleware-stack by returning a new
 * module object, which only works when the module hook intercepts the require().
 */

import { AwsInstrumentation } from "@opentelemetry/instrumentation-aws-sdk";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_BASE_URL = nodeProcess.env.MOCK_LLM_URL!;

async function main() {
  console.log("=== OTel Contrib JS: AWS Bedrock Conformance Test ===");

  // Set up OTel providers
  const otel = setupOtel();
  const { provider, meterProvider } = otel;

  // Register instrumentation BEFORE importing AWS SDK modules.
  // registerInstrumentations installs require-in-the-middle hooks.
  registerInstrumentations({
    tracerProvider: provider,
    meterProvider: meterProvider,
    instrumentations: [new AwsInstrumentation()],
  });

  // Dynamic import AFTER registration so hooks can intercept the AWS SDK's
  // internal require('@smithy/middleware-stack') calls.
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

  // Scenario: basic chat
  console.log("  [chat] basic chat completion");
  const resp = await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [{ role: "user", content: [{ text: "Say hello." }] }],
    })
  );
  console.log(`    -> ${resp.output?.message?.content?.[0]?.text?.slice(0, 60)}`);

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming chat completion");
  const streamResp = await client.send(
    new ConverseStreamCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [{ role: "user", content: [{ text: "Tell me a joke." }] }],
    })
  );
  let text = "";
  if (streamResp.stream) {
    for await (const event of streamResp.stream) {
      if (event.contentBlockDelta?.delta?.text) {
        text += event.contentBlockDelta.delta.text;
      }
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);

  // Scenario: embeddings
  console.log("  [embeddings] Bedrock Titan Embeddings");
  const embResp = await client.send(
    new InvokeModelCommand({
      modelId: "amazon.titan-embed-text-v2:0",
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify({ inputText: "Hello, world!" }),
    })
  );
  const embResult = JSON.parse(new TextDecoder().decode(embResp.body));
  console.log(`    -> embedding dim: ${embResult.embedding.length}`);

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
