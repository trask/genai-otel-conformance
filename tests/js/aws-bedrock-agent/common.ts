/**
 * Shared test infrastructure for AWS Bedrock Agent JS conformance tests.
 */

import { trace, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/api";
import { setupOtel, flushAndShutdownOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL!;
const parsedUrl = new URL(MOCK_BASE_URL);
const SERVER_ADDRESS = parsedUrl.hostname;
const SERVER_PORT = parseInt(parsedUrl.port, 10) || 443;

const tracer = trace.getTracer("gen_ai.client.aws_bedrock");

const AGENT_ID = "MOCK_AGENT_ID";
const AGENT_ALIAS_ID = "MOCK_ALIAS_ID";
const SESSION_ID = "mock-session-001";
const AGENT_NAME = "conformance-test-agent";

export async function runInvokeAgent(): Promise<void> {
  console.log("  [invoke_agent] Bedrock Agent Runtime InvokeAgent");

  const { BedrockAgentRuntimeClient, InvokeAgentCommand } = await import(
    "@aws-sdk/client-bedrock-agent-runtime"
  );
  const { NodeHttpHandler } = await import("@smithy/node-http-handler");

  const client = new BedrockAgentRuntimeClient({
    endpoint: MOCK_BASE_URL,
    region: "us-east-1",
    credentials: { accessKeyId: "mock", secretAccessKey: "mock" },
    requestHandler: new NodeHttpHandler(),
  });

  await tracer.startActiveSpan(
    "invoke_agent",
    { kind: SpanKind.CLIENT },
    async (span: Span) => {
      span.setAttribute("gen_ai.operation.name", "invoke_agent");
      span.setAttribute("gen_ai.provider.name", "aws.bedrock");
      span.setAttribute("gen_ai.agent.id", AGENT_ID);
      span.setAttribute("gen_ai.agent.name", AGENT_NAME);
      span.setAttribute("server.address", SERVER_ADDRESS);
      span.setAttribute("server.port", SERVER_PORT);
      try {
        const resp = await client.send(
          new InvokeAgentCommand({
            agentId: AGENT_ID,
            agentAliasId: AGENT_ALIAS_ID,
            sessionId: SESSION_ID,
            inputText: "Hello, agent!",
          })
        );
        let text = "";
        if (resp.completion) {
          for await (const event of resp.completion) {
            if (event.chunk?.bytes) {
              text += new TextDecoder().decode(event.chunk.bytes);
            }
          }
        }
        console.log(`    -> ${text.slice(0, 60)}`);
      } catch (err) {
        span.setStatus({ code: SpanStatusCode.ERROR, message: String(err) });
        span.setAttribute("error.type", (err as Error).constructor.name);
        throw err;
      } finally {
        span.end();
      }
    }
  );
}

export { setupOtel };
