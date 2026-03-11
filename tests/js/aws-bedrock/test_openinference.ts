/**
 * Conformance test: OpenInference (Arize) JS AWS Bedrock instrumentation.
 */

import { BedrockInstrumentation } from "@arizeai/openinference-instrumentation-bedrock";
import { run } from "./common";

function instrument(bedrockModule: any) {
  const instrumentation = new BedrockInstrumentation();
  instrumentation.manuallyInstrument(bedrockModule);
}

run("OpenInference JS: AWS Bedrock Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
