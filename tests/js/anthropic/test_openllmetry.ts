/**
 * Conformance test: OpenLLMetry (Traceloop) JS Anthropic instrumentation.
 */

import { AnthropicInstrumentation } from "@traceloop/instrumentation-anthropic";
import { run } from "./common";

function instrument(anthropicModule: any) {
  const instrumentation = new AnthropicInstrumentation();
  instrumentation.manuallyInstrument(anthropicModule.default);
}

run("OpenLLMetry JS: Anthropic Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
