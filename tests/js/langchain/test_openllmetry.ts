/**
 * Conformance test: OpenLLMetry (Traceloop) JS LangChain instrumentation.
 *
 * Exercises: chat, chat_streaming, embeddings
 * against a mock OpenAI server.
 */

import { LangChainInstrumentation } from "@traceloop/instrumentation-langchain";

import { flushAndShutdownOtel, setupOtel } from "../otel";

const MOCK_BASE_URL = process.env.MOCK_LLM_URL! + "/v1";

async function main() {
  console.log("=== OpenLLMetry JS: LangChain Conformance Test ===");

  const otel = setupOtel();

  // Dynamic import AFTER OTel setup so manuallyInstrument can patch prototypes.
  // tsx (ESM) does not support require-in-the-middle, so enable() does not work.
  const callbackManagerModule = await import("@langchain/core/callbacks/manager");
  const { ChatOpenAI } = await import("@langchain/openai");
  const { StringOutputParser } = await import("@langchain/core/output_parsers");
  const { ChatPromptTemplate } = await import("@langchain/core/prompts");

  const instrumentation = new LangChainInstrumentation();
  instrumentation.manuallyInstrument({ callbackManagerModule });

  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    apiKey: "mock-key",
    configuration: { baseURL: MOCK_BASE_URL },
  });

  // Use RunnableSequence (pipe) so LangChain instrumentation can capture spans
  const prompt = ChatPromptTemplate.fromMessages([["user", "{input}"]]);
  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  console.log("  [chat] basic chat completion via chain");
  const resp = await chain.invoke({ input: "Say hello." });
  console.log(`    -> ${resp.slice(0, 60)}`);

  console.log("  [chat_streaming] streaming chat completion via chain");
  let text = "";
  const stream = await chain.stream({ input: "Tell me a joke." });
  for await (const chunk of stream) {
    text += chunk;
  }
  console.log(`    -> ${text.slice(0, 60)}`);

  console.log("  [embeddings] embedding generation");
  const { OpenAIEmbeddings } = await import("@langchain/openai");
  const embedModel = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: "mock-key",
    configuration: { baseURL: MOCK_BASE_URL },
  });
  const embResult = await embedModel.embedQuery("Hello, world!");
  console.log(`    -> embedding dim: ${embResult.length}`);

  await flushAndShutdownOtel(otel);
}

main().catch((e) => { console.error(e); process.exit(1); });
