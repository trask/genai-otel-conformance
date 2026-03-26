/**
 * Conformance test: Reference instrumentation for Vertex AI.
 */
import { trace } from "@opentelemetry/api";
import { flushAndShutdownOtel, setupOtel } from "../otel";

const nodeProcess = globalThis.process as NodeJS.Process & {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

const MOCK_LLM_URL = nodeProcess.env.MOCK_LLM_URL!;
const tracer = trace.getTracer("gen_ai.reference");

// Patch fetch before any SDK imports to intercept auth and redirect HTTPS→HTTP.
const _originalFetch = globalThis.fetch;
globalThis.fetch = async function (input: any, init?: any) {
  let url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;

  // Mock OAuth token endpoint so the SDK gets a fake access token
  if (url.includes("oauth2.googleapis.com") || url.includes("/token")) {
    return new Response(
      JSON.stringify({
        access_token: "mock-token",
        token_type: "Bearer",
        expires_in: 3600,
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }

  // Redirect HTTPS to HTTP for mock server
  url = url.replace(/^https:\/\//, "http://");
  return _originalFetch(url, init);
};

async function main() {
  console.log("=== Reference: Vertex AI JS Conformance Test ===");
  const otel = setupOtel();

  const { VertexAI } = await import("@google-cloud/vertexai");
  const { createFakeGoogleAuth } = await import(
    "@google-cloud/vertexai/build/src/testing/fake_google_auth"
  );
  const mockUrl = new URL(MOCK_LLM_URL);
  const vertexai = new VertexAI({
    project: "test-project",
    location: "us-central1",
    apiEndpoint: `${mockUrl.hostname}:${mockUrl.port}`,
  });
  (vertexai as any).googleAuth = createFakeGoogleAuth({
    accessToken: "mock-token",
  });
  const requestModel = "gemini-2.0-flash";
  const model = vertexai.getGenerativeModel({ model: requestModel });

  // Scenario: chat
  console.log("  [chat] basic content generation (reference)");
  await tracer.startActiveSpan("chat gemini-2.0-flash", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "vertex_ai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const result = await model.generateContent("Say hello.");
    const resp = result.response;
    const responseModel = (resp as any).modelVersion ?? requestModel;
    span.setAttribute("gen_ai.response.model", responseModel);
    const candidate = resp.candidates?.[0];
    if (candidate?.finishReason) {
      span.setAttribute("gen_ai.response.finish_reasons", [candidate.finishReason]);
    }
    if (resp.usageMetadata) {
      if (resp.usageMetadata.promptTokenCount) {
        span.setAttribute("gen_ai.usage.input_tokens", resp.usageMetadata.promptTokenCount);
      }
      if (resp.usageMetadata.candidatesTokenCount) {
        span.setAttribute("gen_ai.usage.output_tokens", resp.usageMetadata.candidatesTokenCount);
      }
    }
    const text = candidate?.content?.parts?.[0]?.text ?? "";
    console.log(`    -> ${text.slice(0, 60)}`);
    span.end();
  });

  // Scenario: streaming chat
  console.log("  [chat_streaming] streaming content generation (reference)");
  await tracer.startActiveSpan("chat gemini-2.0-flash", async (span) => {
    span.setAttribute("gen_ai.operation.name", "chat");
    span.setAttribute("gen_ai.provider.name", "vertex_ai");
    span.setAttribute("gen_ai.request.model", requestModel);
    const result = await model.generateContentStream("Tell me a joke.");
    let text = "";
    for await (const chunk of result.stream) {
      const part = chunk.candidates?.[0]?.content?.parts?.[0]?.text;
      if (part) text += part;
    }
    const aggregated = await result.response;
    const responseModel = (aggregated as any).modelVersion ?? requestModel;
    span.setAttribute("gen_ai.response.model", responseModel);
    const candidate = aggregated.candidates?.[0];
    if (candidate?.finishReason) {
      span.setAttribute("gen_ai.response.finish_reasons", [candidate.finishReason]);
    }
    if (aggregated.usageMetadata) {
      if (aggregated.usageMetadata.promptTokenCount) {
        span.setAttribute("gen_ai.usage.input_tokens", aggregated.usageMetadata.promptTokenCount);
      }
      if (aggregated.usageMetadata.candidatesTokenCount) {
        span.setAttribute("gen_ai.usage.output_tokens", aggregated.usageMetadata.candidatesTokenCount);
      }
    }
    console.log(`    -> ${text.slice(0, 60)}`);
    span.end();
  });

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); nodeProcess.exit(1); });
