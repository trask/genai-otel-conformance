# Validates the JSON payload of GenAI content attributes against the
# semconv JSON schemas.  Schema constants (_schema_*) are defined in
# _schemas.rego, which is generated at test-run time from the semconv
# repository (docs/gen-ai/*.json) and placed alongside this file.
# Weaver only loads .rego files from --advice-policies, so schemas are
# inlined as Rego constants rather than loaded as OPA data documents.

package live_check_advice

import rego.v1

_genai_content_schemas := {
	"gen_ai.input.messages":      _schema_input_messages,
	"gen_ai.output.messages":     _schema_output_messages,
	"gen_ai.system_instructions": _schema_system_instructions,
}

deny contains result if {
	input.sample.attribute
	attr_name := input.sample.attribute.name
	attr_value := input.sample.attribute.value
	is_string(attr_value)

	schema := _genai_content_schemas[attr_name]

	parsed := json.unmarshal(attr_value)

	[matched, errors] := json.match_schema(parsed, schema)
	not matched

	result := {
		"type":         "advice",
		"advice_type":  "genai_content_schema",
		"advice_level": "violation",
		"context": {
			"attribute": input.sample.attribute.name,
			"errors":    errors,
		},
		"message": sprintf(
			"Attribute '%v' value does not conform to the GenAI schema: %v",
			[input.sample.attribute.name, errors],
		),
	}
}
