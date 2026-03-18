# PR Activity Comparison: OpenLLMetry vs OTel Python Contrib (GenAI)

**Period:** September 18, 2025 – March 18, 2026 (6 months)
**Generated:** March 18, 2026

Compares PR activity between:
- [traceloop/openllmetry](https://github.com/traceloop/openllmetry) (full repo)
- [open-telemetry/opentelemetry-python-contrib](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation-genai) (`instrumentation-genai/` directory only)

> Renovate, Dependabot, CodeRabbitAI, and otelbot PRs are excluded from all counts.

---

## Summary

| Metric | OpenLLMetry | OTel Python Contrib (GenAI) |
|---|---|---|
| Total PRs | 150 | 109 |
| Merged | 74 (49%) | 59 (54%) |
| Closed (not merged) | 25 | 25 |
| Still open | 51 | 25 |
| Unique authors | 57 | 32 |
| Repeat contributors (3+ PRs) | 13 | 11 |
| Single-PR contributors | 40 | 13 |
| Median time to merge | 21 hours (0.9 days) | 76 hours (3.2 days) |
| Average time to merge | 152 hours (6.3 days) | 227 hours (9.5 days) |

## Monthly Breakdown

### OpenLLMetry

| Month | Created | Merged |
|---|---|---|
| 2025-09 | 4 | 0 |
| 2025-10 | 15 | 6 |
| 2025-11 | 27 | 21 |
| 2025-12 | 28 | 15 |
| 2026-01 | 22 | 10 |
| 2026-02 | 34 | 17 |
| 2026-03 | 22 | 5 |

### OTel Python Contrib (GenAI)

| Month | Created | Merged |
|---|---|---|
| 2025-09 | 8 | 6 |
| 2025-10 | 29 | 19 |
| 2025-11 | 9 | 8 |
| 2025-12 | 22 | 16 |
| 2026-01 | 9 | 8 |
| 2026-02 | 37 | 19 |
| 2026-03 | 15 | 3 |

## Top Contributors (by merged PRs)

### OpenLLMetry

| Author | Merged | Total | Notes |
|---|---|---|---|
| nina-kollman | 16 | 27 | Traceloop SDK features (evals, datasets, decorators) |
| galkleinman | 15 | 16 | Dependency management, semconv, new instrumentations |
| nirga | 9 | 9 | 100% merge rate |
| elinacse | 7 | 13 | Bug fixes across OpenAI, Anthropic, Langchain, Watsonx |
| OzBenSimhonTraceloop | 3 | 3 | CI/release infrastructure |
| EliJaghab | 3 | 5 | OpenAI Agents realtime session fixes |
| dinmukhamedm | 3 | 4 | Dependency updates, bug fixes |
| duanyutong | 3 | 3 | |

### OTel Python Contrib (GenAI)

| Author | Merged | Total | Notes |
|---|---|---|---|
| xrmx | 11 | 12 | Test infra, lint, boilerplate cleanup |
| DylanRussell | 8 | 9 | Google GenAI token counting, streaming fixes |
| nagkumar91 | 8 | 17 | OpenAI Agents, streaming, memory operations |
| aabmass | 7 | 10 | Release prep, util-genai, cross-cutting fixes |
| vasantteja | 6 | 9 | Anthropic streaming, agents boilerplate, OpenAI Responses |
| wrisa | 3 | 5 | Langchain instrumentation |
| wikaaaaa | 3 | 5 | Tool definitions, semconv alignment |
| emdneto | 2 | 2 | Python 3.14 support, CI improvements |

## Key Observations

### Volume and Velocity
- **OpenLLMetry receives ~40% more PRs** (150 vs 109), which is expected given it covers many more provider instrumentations (20+ packages) compared to the focused GenAI subset of otel-python-contrib.
- **OTel Python Contrib has a higher merge rate** (54% vs 49%), suggesting tighter contributor alignment or more targeted submissions.
- **OpenLLMetry merges significantly faster** — median 0.9 days vs 3.2 days. This likely reflects the difference between a company-maintained project (Traceloop) that can merge quickly vs. an open-source foundation project with formal review processes.

### Contributor Profiles
- **OpenLLMetry has more one-off contributors** (40 single-PR authors vs 13). Many of these are community bug fixes or feature additions that were not merged, indicating a wider but shallower contributor base.
- **OTel Python Contrib has a more concentrated contributor base** with a core group (xrmx, DylanRussell, nagkumar91, aabmass, vasantteja) driving the majority of merged work.
- **OpenLLMetry's top contributors are predominantly Traceloop employees** (nina-kollman, galkleinman, nirga, OzBenSimhonTraceloop, elinacse), with community contributions making up a smaller share of merged work.
- **OTel Python Contrib contributors are more diverse by affiliation** — Google (DylanRussell, aabmass), community volunteers (xrmx, vasantteja, nagkumar91), and Microsoft (lmolkova).

### Nature of Work
- **OpenLLMetry** work is heavily focused on:
  - Bug fixes across existing provider instrumentations (OpenAI, Anthropic, Langchain, etc.)
  - Traceloop SDK features (evals, datasets, conversation decorators)
  - Dependency/semconv version bumps and new instrumentations (Voyage AI)
  - Recent push toward OTel GenAI semconv compliance (several open PRs from max-deygin-traceloop)

- **OTel Python Contrib (GenAI)** work is focused on:
  - Adding new provider instrumentations (Google GenAI, Anthropic, OpenAI Agents, Langchain)
  - Streaming support and response wrapper infrastructure
  - GenAI semantic conventions alignment and util-genai shared library
  - Release engineering and CI improvements
  - OpenAI Responses API support (newer feature)

### Open PR Backlog
- **OpenLLMetry has 51 open PRs**, notably including a coordinated effort to align instrumentations with OTel GenAI semantic conventions (5 PRs from max-deygin-traceloop).
- **OTel Python Contrib has 25 open PRs**, including GenAI utils/agent metrics work and several new instrumentation features.

### Convergence Signal
- Both projects are converging on OTel GenAI semantic conventions. OpenLLMetry is visibly working to align its instrumentation output with the OTel standard (open PRs for semconv migration, anthropic compliance, openai-agents compliance). This suggests awareness that the OTel-native instrumentation is becoming the reference.
