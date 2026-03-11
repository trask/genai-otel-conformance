/**
 * Conformance test: OpenLLMetry (Traceloop) JS AWS Bedrock instrumentation.
 */

import { BedrockInstrumentation } from "@traceloop/instrumentation-bedrock";
import { run } from "./common";

function instrument(bedrockModule: any) {
  const instrumentation = new BedrockInstrumentation();
  instrumentation.manuallyInstrument(bedrockModule);
}

run("OpenLLMetry JS: AWS Bedrock Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
