/**
 * Shared test infrastructure for Vertex AI JS conformance tests.
 *
 * The Vertex AI SDK uses HTTPS and Google Auth. For mock testing we:
 * 1. Override globalThis.fetch to redirect HTTPS to HTTP
 * 2. Mock OAuth token endpoint responses
 * 3. Use authorized_user credentials so the SDK attempts a token refresh
 *    (intercepted by our fetch override)
 */

import type { GenerativeModel } from "@google-cloud/vertexai";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_LLM_URL = process.env.MOCK_LLM_URL!;

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

export async function runChat(model: GenerativeModel) {
  console.log("  [chat] basic content generation");
  const result = await model.generateContent("Say hello.");
  const text = result.response.candidates[0].content.parts[0].text;
  console.log(`    -> ${text.slice(0, 60)}`);
}

export async function runChatStreaming(model: GenerativeModel) {
  console.log("  [chat_streaming] streaming content generation");
  const result = await model.generateContentStream("Tell me a joke.");
  let text = "";
  for await (const chunk of result.stream) {
    const part = chunk.candidates?.[0]?.content?.parts?.[0]?.text;
    if (part) {
      text += part;
    }
  }
  console.log(`    -> ${text.slice(0, 60)}`);
}

export async function runChatToolCall(model: GenerativeModel) {
  console.log("  [chat_tool_call] content generation with tool calling");
  const result = await model.generateContent({
    contents: [{ role: "user", parts: [{ text: "What's the weather in Seattle?" }] }],
    tools: [
      {
        functionDeclarations: [
          {
            name: "get_weather",
            description: "Get the current weather",
            parameters: {
              type: "OBJECT" as any,
              properties: {
                location: { type: "STRING" as any, description: "City name" },
              },
              required: ["location"],
            },
          },
        ],
      },
    ],
  });
  const part = result.response.candidates?.[0]?.content?.parts?.[0];
  if (part && "functionCall" in part && part.functionCall) {
    console.log(`    -> tool_call: ${part.functionCall.name}`);
  } else {
    const text = part?.text ?? "";
    console.log(`    -> ${text.slice(0, 60)}`);
  }
}

export async function run(title: string, instrumentFn: (vertexaiModule: any) => void) {
  console.log(`=== ${title} ===`);

  const otel = setupOtel();

  // Import @google-cloud/vertexai and pass to instrument function for manual patching.
  // ESM + tsx does not support require-in-the-middle hooks, so we use
  // manuallyInstrument() to directly patch the module prototypes.
  const vertexaiModule = await import("@google-cloud/vertexai");
  instrumentFn(vertexaiModule);

  const { VertexAI } = vertexaiModule;
  // Use the SDK's built-in FakeGoogleAuth to avoid real OAuth token refresh
  // (the auth library uses gaxios/node-fetch, not globalThis.fetch, so our
  // fetch override cannot intercept it).
  const { createFakeGoogleAuth } = await import("@google-cloud/vertexai/build/src/testing/fake_google_auth");
  const mockUrl = new URL(MOCK_LLM_URL);
  const vertexai = new VertexAI({
    project: "test-project",
    location: "us-central1",
    apiEndpoint: `${mockUrl.hostname}:${mockUrl.port}`,
  });
  (vertexai as any).googleAuth = createFakeGoogleAuth({ accessToken: "mock-token" });

  const model = vertexai.getGenerativeModel({ model: "gemini-2.0-flash" });

  await runChat(model);
  await runChatStreaming(model);
  await runChatToolCall(model);

  await flushAndShutdownOtel(otel);
}
