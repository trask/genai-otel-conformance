/**
 * Conformance test: OpenLLMetry (Traceloop) JS Vertex AI instrumentation.
 */

import { VertexAIInstrumentation } from "@traceloop/instrumentation-vertexai";
import { run } from "./common";

function instrument(vertexaiModule: any) {
  const instrumentation = new VertexAIInstrumentation();
  instrumentation.manuallyInstrument(vertexaiModule);
}

run("OpenLLMetry JS: Vertex AI Conformance Test", instrument).catch((e) => { console.error(e); process.exit(1); });
