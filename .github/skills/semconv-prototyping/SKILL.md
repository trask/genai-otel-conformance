---
name: semconv-prototyping
description: 'Prototype and review semantic convention attributes, especially manual instrumentation examples for native library instrumentation. Use when evaluating whether an attribute is directly observable, semantically derivable by native instrumentation, or too app-specific, cross-call, or test-only to be convincing.'
argument-hint: 'Describe the proposed semantic convention or manual instrumentation example to review.'
---

# Semantic Convention Prototyping

Use this skill when reviewing or drafting semantic convention prototypes in this repository, especially when judging whether a manual instrumentation example is believable for native instrumentation.

## Core Rule

Do not ask only whether the attribute is a literal field copy.

Ask whether native instrumentation for the underlying library can populate it correctly and consistently from information the library already owns.

## Review Buckets

Classify each candidate attribute as one of:

### 1. Directly Observable

The instrumentation can read it from the current call boundary.

Typical sources:
- method arguments
- return values
- exceptions
- client configuration
- current request or response objects

### 2. Semantically Derivable

The instrumentation can compute it from library-owned semantics without app-specific guesswork.

This includes normalized values that are not literal field copies, as long as the derivation is stable and grounded in the library contract.

### 3. Too Weak

Flag it if it depends on app-specific naming, opaque identifiers, cached data from another call, test-only scaffolding, or guessing a semantic enum from arbitrary strings.

## Questions To Ask Per Attribute

1. What exact object, argument, response, or exception would native instrumentation read this from?
2. If the value is derived, is the derivation defined by the library API or just by the test setup?
3. Would the same instrumentation logic work outside this conformance test with real application inputs?
4. Does populating the attribute require remembering data from a previous call rather than the current operation?
5. Is the value derived from an opaque string that only looks meaningful in this test?

If questions 1 through 3 are strong and 4 through 5 are no, the attribute is usually fine.

## Quick Checks

- Good signs: current-call ids, result counts, and values already present on the current request, response, exception, or library object.
- Acceptable derivation: normalized scope, content, or expiration values only when the library contract makes the meaning explicit and the derivation is stable.
- Weak signs: carried-forward ids from earlier calls, semantics inferred from opaque strings like `test-user-001`, flattened test-specific summaries, or values that depend on setup state rather than the current operation.

## How To Use In This Repo

When reviewing a manual scenario or prototype:

1. List each attribute set on the span.
2. Mark each attribute as `direct`, `derivable`, or `weak`.
3. Prefer keeping `direct` and `derivable` attributes.
4. Remove or downgrade `weak` attributes from manual examples and expected baselines.
5. When in doubt, explain the exact native instrumentation mechanism that would populate the field.

## Output Format

When using this skill in a review, summarize the result in three groups.

- `Directly observable`
- `Semantically derivable`
- `Too app-specific or cross-call`

For each flagged attribute, state why it is weak and point to the current call inputs or outputs that are missing the needed information.
