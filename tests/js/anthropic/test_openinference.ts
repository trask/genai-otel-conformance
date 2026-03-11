/**
 * Conformance test: OpenInference (Arize) JS Anthropic instrumentation.
 */

import { AnthropicInstrumentation } from "@arizeai/openinference-instrumentation-anthropic";
import { run } from "./common";

function instrument(anthropicModule: any) {
  const instrumentation = new AnthropicInstrumentation();
  instrumentation.manuallyInstrument(anthropicModule);
}

run("OpenInference JS: Anthropic Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
