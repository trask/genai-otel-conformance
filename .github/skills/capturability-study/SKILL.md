---
name: capturability-study
description: 'Use when reviewing capturability of proposed semantic convention attributes against manual instrumentation examples, especially in this repository. Determines whether an attribute is directly observable, semantically derivable by native instrumentation, too app-specific, or a capture gap relative to the upstream proposal while preserving the proposal in the local capturability study.'
argument-hint: 'Describe the proposed attribute, span, or manual instrumentation example to evaluate for capturability.'
---

# Semantic Convention Capturability Study

Use this skill when reviewing proposed semantic convention attributes against manual instrumentation examples in this repository, especially when judging whether a value is credibly capturable by native instrumentation.

## Non-Goal

This skill is not for arguing that the upstream proposal is correct or incorrect.

Its job is narrower: determine whether the current example honestly shows what a provider can emit while the proposal stays fixed.

## Capturability Study Stance

In this repository, a mismatch between a proposed semantic convention and a believable manual example is not automatically a bug in the example.

It may indicate a capture gap between the current proposal and what a provider can credibly emit.

When this repo is being used as a capturability study for an upstream semantic convention proposal:

- Default to preserving the current upstream proposal in the local capturability study.
- The normal job of the capturability study is to show which attributes can be captured and which cannot while still mirroring the proposal's current requirement levels.
- Do not assume missing attributes should be added just to satisfy the proposal.
- Treat non-capturable attributes as documented capture gaps relative to the upstream proposal.
- Distinguish `example needs fixing` from `provider does not demonstrate this attribute`.
- Do not frame the capturability study itself as inherently supporting or opposing the upstream proposal. It is primarily documenting capturability against the proposal as written.

## Capturability Study Mode

Unless the user explicitly asks for a different goal, evaluate the capturability study in this order:

1. Does the local repo accurately mirror the current upstream proposal?
2. Does each provider example honestly show what that provider can and cannot capture?
3. If a provider cannot capture a proposed attribute, should the local example stay as a documented capture gap instead of being changed?

In this mode, the default recommendation is not "change the requirement level here".

The default recommendation is one of:

- `fix the example` when the current call boundary really exposes the value
- `leave example unchanged; provider does not demonstrate this attribute` when the provider cannot credibly capture the value

## Core Rule

Do not ask only whether the attribute is a literal field copy.

Ask whether native instrumentation for the underlying library can populate it correctly and consistently from information the library already owns.

If you cannot name the concrete argument, object, response field, exception, or library-owned state that would produce the value, treat the attribute as not credibly capturable.

## Attribute Classes

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

## Review Procedure

When reviewing a manual scenario or example:

1. List each attribute set on the span.
2. Mark each attribute as `direct`, `derivable`, or `weak`.
3. First ask whether the local capturability study is trying to mirror the upstream proposal as written. If yes, do not treat every unsupported attribute as a local spec bug.
4. For each weak or missing attribute, decide whether:
	- the example should be fixed because the SDK call already exposes the data
	- the example should remain unchanged and the provider should be treated as not demonstrating that attribute
	
	Prefer these in order: `fix example` -> `leave unchanged; provider does not demonstrate attribute`.
5. Do not recommend changing the example unless the current call boundary actually provides the needed information.
6. When in doubt, explain the exact native instrumentation mechanism that would populate the field, or explicitly state that no credible mechanism is apparent.
7. If the repository is acting as a capturability study for an upstream proposal, prefer preserving honest capture gaps over making the local example superficially conform.

## Do Not Conflate

Keep these judgments separate:

- `capturability study mirrors upstream proposal correctly`
- `provider example supports this attribute`
- `provider example does not support this attribute`

A correct review can say all of the following at once:

- the local capturability study should keep the current requirement level
- this provider should not be changed to fake compliance
- this provider does not demonstrate the attribute

## Output Format

When using this skill in a review, summarize the result in four groups.

- `Directly observable`
- `Semantically derivable`
- `Too app-specific or cross-call`
- `Study recommendation`

For each flagged attribute, state:

- why it is weak
- the exact current-call input, output, response field, exception, or library-owned state that would be needed to populate it
- whether that source is actually available in the example

Under `Study recommendation`, state one of:

- `fix local example`
- `leave example unchanged; provider does not demonstrate this attribute`
