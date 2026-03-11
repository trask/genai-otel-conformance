/**
 * Conformance test: OpenLLMetry (Traceloop) JS Cohere instrumentation.
 */

import { CohereInstrumentation } from "@traceloop/instrumentation-cohere";
import { run } from "./common";

function instrument(cohereModule: any) {
  const instrumentation = new CohereInstrumentation();
  instrumentation.manuallyInstrument(cohereModule);
}

run("OpenLLMetry JS: Cohere Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
