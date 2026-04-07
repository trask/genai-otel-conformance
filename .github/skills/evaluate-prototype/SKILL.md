---
name: evaluate-prototype
description: 'Use when reviewing completed prototype instrumentation, conformance test outputs, or data-prototype results against a semantic-conventions PR, upstream proposal, or spec change. Evaluates prototype coverage, capturability, direct observability, derivable fields, missing supporting libraries, and honest capture gaps without forcing fake compliance.'
argument-hint: 'Describe the semantic-conventions PR and the prototype implementation, files, or results to evaluate.'
---

# Prototype Evaluation

Use this skill after prototype instrumentation has been added for a semantic-conventions PR and the repository needs an evaluation of coverage quality and capturability.

## Goal

Determine whether the resulting prototype honestly shows what each library can emit while mirroring the semantic-conventions PR as written.

## Non-Goal

This skill is not for arguing that the upstream proposal is correct or incorrect.

Its job is to evaluate the resulting prototype set:

- which libraries genuinely demonstrate the proposed behavior
- which emitted fields are direct, derivable, weak, or unsupported
- which missing fields are honest capture gaps rather than implementation bugs
- which supporting libraries still need prototype coverage

## Evaluation Stance

In this repository, a mismatch between a proposed semantic convention and a believable prototype is not automatically a bug in the prototype.

It may indicate a real capture gap between the proposal and what native instrumentation for that library can credibly emit.

Default to these principles:

- Preserve the semantic-conventions PR as the evaluation target.
- Judge each library on what its current call boundary honestly exposes.
- Distinguish `prototype needs fixing` from `library does not demonstrate this field`.
- Prefer honest capture gaps over superficial compliance.
- Evaluate coverage across all supporting libraries, not just the first one that passes.

## Core Rule

Do not ask only whether an attribute appears in the prototype output.

Ask whether native instrumentation for the underlying library can populate it correctly and consistently from information the library already owns.

If you cannot name the concrete argument, object, response field, streamed event, exception, or library-owned state that would produce the value, treat the field as not credibly demonstrated.

## Attribute Classes

Classify each candidate field as one of:

### 1. Directly Observable

The instrumentation can read it from the current call boundary.

Typical sources:
- method arguments
- return values
- streamed chunks or events
- exceptions
- client configuration
- current request or response objects

### 2. Semantically Derivable

The instrumentation can compute it from library-owned semantics without app-specific guesswork.

This includes normalized values that are not literal field copies, as long as the derivation is stable and grounded in the library contract.

### 3. Too Weak Or App-Specific

Flag it if it depends on app-specific naming, opaque identifiers, cached data from another call, test-only scaffolding, or guessing a semantic enum from arbitrary strings.

## Coverage Questions

1. Does the prototype set mirror the semantic-conventions PR accurately?
2. Which libraries in this repository support the affected operation?
3. Which of those libraries were prototyped?
4. For each prototype, what exact current-call source backs every emitted field?
5. Are any fields missing even though the current library boundary exposes them?
6. Are any emitted fields weak because they rely on cross-call memory or test-only assumptions?

## Review Procedure

1. Read the semantic-conventions PR and reduce it to the spans and attributes that should be evaluated.
2. List the libraries in this repository that support the affected operation.
3. Confirm which of those libraries received prototype updates.
4. For each updated library, list each emitted span and attribute.
5. Mark each attribute as `direct`, `derivable`, or `weak`.
6. For each missing or weak field, decide whether:
   - the prototype should be fixed because the SDK call already exposes the data
   - the prototype should remain unchanged because the library does not credibly demonstrate the field
7. For each library that was not prototyped, decide whether it is out of scope, not yet implemented here, or a missed supporting library.
8. Prefer these outcomes in order: `fix prototype` -> `add missing supporting library prototype` -> `leave unchanged; honest capture gap`.

## Determining the PR Changeset

Use `gh pr diff <number>` (not `git diff main`) to get the changeset. A stale local `main` causes `git diff main` to include unrelated commits.

## Do Not Conflate

Keep these judgments separate:

- `semantic-conventions PR mirrored correctly`
- `library prototype supports this field`
- `library prototype does not support this field`
- `supporting library was never prototyped`

A correct evaluation can say all of the following at once:

- the repository is targeting the right proposal
- one library prototype should be fixed
- another library honestly cannot emit the field
- a third supporting library still needs prototype coverage

## Output Format

When using this skill in a review, summarize the result in five groups.

- `Credible prototype coverage`
- `Semantically derivable fields`
- `Too app-specific or cross-call`
- `Missing supporting library prototypes`
- `Evaluation recommendation`

For each flagged field or missing library, state:

- why it is weak, missing, or incomplete
- the exact current-call source that would be needed to support it
- whether that source is actually available in the library example

Under `Evaluation recommendation`, use one or more of:

- `fix prototype`
- `add prototype for supporting library`
- `leave unchanged; honest capture gap`