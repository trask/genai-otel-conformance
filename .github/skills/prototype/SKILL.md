---
name: prototype
description: 'Use when implementing a semantic-conventions PR, upstream proposal diff, spec change, or new GenAI span or attribute in this repository. Adds prototype instrumentation, conformance test updates, inline attribute emission, and data-prototype coverage for every Python, JS, Java, and .NET library that credibly supports the change.'
argument-hint: 'Describe the semantic-conventions PR or the convention changes that need prototype coverage.'
---

# Prototype Coverage

Use this skill when a semantic-conventions PR introduces or changes GenAI spans, attributes, or requirement levels and the repository needs prototype coverage across all libraries that support the new behavior.

## Goal

Turn a semantic-conventions PR into concrete prototype implementations in this repository.

The result should be a set of prototype scenarios and emitted attributes that honestly exercise every supporting library without faking values that the library cannot credibly expose.

## Non-Goals

This skill is not for deciding whether the upstream proposal is correct.

This skill is also not the final evaluation pass. After adding the prototypes, use the `evaluate-prototype` skill to judge capturability, coverage quality, and honest capture gaps.

## Core Stance

- Start from the semantic-conventions PR as written.
- Add prototypes for every library in this repository that supports the affected operation and can credibly expose the new fields at the current call boundary.
- Do not skip a supported library just because the implementation is repetitive.
- Do not force unsupported libraries to emit guessed, hardcoded, cross-call, or app-specific values.
- Prefer broad, consistent prototype coverage across ecosystems when the same library behavior exists.

## What Counts As Supporting The PR

A library should usually get a prototype update when all of the following are true:

1. The repository already has a test scenario for the relevant operation, or the operation can be added naturally within that library's existing conformance structure.
2. The library API or current response objects expose the information needed for the new span or attribute.
3. The prototype can emit the value from the current request, current response, current exception, or stable library-owned state.

If the value would have to be guessed, carried forward from an unrelated call, or synthesized from test-only scaffolding, do not force it into the prototype.

## Implementation Rules

When editing prototype tests in this repository:

- Emit attributes inline at the span or activity site.
- Keep request, derived, and response attributes close together.
- Reuse the same current-call variable that the SDK call uses when emitting request attributes.
- Read response values from the current response or streamed result object.
- Avoid helpers that hide emitted attributes.
- Prefer simple, explicit instrumentation over abstractions.

## Procedure

1. Read the semantic-conventions PR and extract the exact changed spans, attributes, requirement levels, and examples.
2. Translate the PR into a concrete implementation worklist grouped by operation, not by prose section.
3. Inventory the libraries in this repository that implement the affected operation across Python, JS, Java, and .NET.
4. For each library, decide whether the changed fields are credibly available from the current call boundary.
5. Add or update the prototype scenario for every supporting library.
6. Emit the new prototype attributes inline and keep them tied to current request or response values.
7. Update the corresponding prototype outputs such as `data-prototype.json` and any generated result artifacts required by the repo workflow.
8. Keep unsupported libraries honest. If a library cannot credibly emit a field, leave it out and record that it will need evaluation as a capture gap.
9. Run targeted validation for the changed libraries when feasible.

## Coverage Expectations

The default expectation is repository-wide prototype coverage for all supporting libraries, not a single illustrative example.

When the same semantic-convention change applies to multiple ecosystems, look for parallel implementations instead of stopping after the first passing library.

## Output Format

When using this skill, summarize the work in four groups.

- `PR changes`
- `Libraries updated`
- `Libraries not updated`
- `Validation`

Under `Libraries not updated`, state whether each library is:

- `not applicable`
- `not yet implemented in this repo`
- `honest capture gap; evaluate separately`

If any library was intentionally left without a prototype, explain the exact missing current-call source that prevented a credible implementation.