/**
 * Conformance test: OpenLLMetry (Traceloop) JS OpenAI instrumentation.
 */

import { OpenAIInstrumentation } from "@traceloop/instrumentation-openai";
import { run } from "./common";

function instrument(openaiModule: any) {
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.manuallyInstrument(openaiModule.default);
}

run("OpenLLMetry JS: OpenAI Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
