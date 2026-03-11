/**
 * Conformance test: OpenInference (Arize) JS OpenAI instrumentation.
 */

import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import { run } from "./common";

function instrument(openaiModule: any) {
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.manuallyInstrument(openaiModule);
}

run("OpenInference JS: OpenAI Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
