# PR Activity Comparison: OpenLLMetry vs OTel Python Contrib (GenAI)

**Period:** March 18, 2025 – March 18, 2026 (12 months)
**Generated:** March 18, 2026

Compares PR activity between:
- [traceloop/openllmetry](https://github.com/traceloop/openllmetry) (full repo)
- [open-telemetry/opentelemetry-python-contrib](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation-genai) (`instrumentation-genai/` directory only)

> Renovate, Dependabot, CodeRabbitAI, and otelbot PRs are excluded from all counts.

---

## Summary

| Metric | OpenLLMetry | OTel Python Contrib (GenAI) |
|---|---|---|
| Total PRs | 384 | 173 |
| Merged | 241 (63%) | 101 (58%) |
| Closed (not merged) | 58 | 38 |
| Still open | 85 | 34 |
| Unique authors | 96 | 47 |
| Repeat contributors (3+ PRs) | 25 | 15 |
| Single-PR contributors | 63 | 24 |
| Median time to merge | 22 hours (0.9 days) | 120 hours (5.0 days) |
| Average time to merge | 217 hours (9.1 days) | 454 hours (18.9 days) |

## Monthly Breakdown

### OpenLLMetry

| Month | Created | Merged |
|---|---|---|
| 2025-03 | 7 | 4 |
| 2025-04 | 19 | 12 |
| 2025-05 | 37 | 16 |
| 2025-06 | 33 | 16 |
| 2025-07 | 53 | 51 |
| 2025-08 | 65 | 53 |
| 2025-09 | 24 | 12 |
| 2025-10 | 14 | 8 |
| 2025-11 | 26 | 21 |
| 2025-12 | 28 | 12 |
| 2026-01 | 22 | 11 |
| 2026-02 | 34 | 17 |
| 2026-03 | 22 | 8 |

### OTel Python Contrib (GenAI)

| Month | Created | Merged |
|---|---|---|
| 2025-03 | 4 | 2 |
| 2025-04 | 5 | 4 |
| 2025-05 | 4 | 2 |
| 2025-06 | 7 | 0 |
| 2025-07 | 9 | 2 |
| 2025-08 | 8 | 4 |
| 2025-09 | 18 | 14 |
| 2025-10 | 25 | 16 |
| 2025-11 | 9 | 11 |
| 2025-12 | 18 | 10 |
| 2026-01 | 7 | 7 |
| 2026-02 | 36 | 17 |
| 2026-03 | 23 | 12 |

## Top Contributors (by merged PRs)

### OpenLLMetry

| Author | Merged | Total | Notes |
|---|---|---|---|
| galkleinman | 42 | 51 | Dependency management, semconv, new instrumentations (Voyage AI), CI/uv migration |
| nirga | 40 | 48 | Agno framework support, MCP fixes, structured outputs, SDK fixes |
| LuizDMM | 23 | 24 | should_send_prompts checks, event emission for VertexAI/LlamaIndex/Bedrock, semconv alignment |
| nina-kollman | 20 | 39 | Traceloop SDK features (evals, datasets, conversation decorator, experiments) |
| dinmukhamedm | 19 | 23 | Dependency updates, Cohere v2, Anthropic fixes, OpenAI compatibility |
| minimAluminiumalism | 12 | 15 | Token usage improvements, non-consumed stream support, LlamaParse instrumentation |
| ronensc | 9 | 9 | 100% merge rate — CrewAI, Milvus, LangChain/LangGraph fixes |
| elinacse | 7 | 14 | Bug fixes across OpenAI, Anthropic, Langchain, Watsonx, Ollama, Qdrant |

### OTel Python Contrib (GenAI)

| Author | Merged | Total | Notes |
|---|---|---|---|
| xrmx | 17 | 18 | Test infra, lint, raw_response streaming, choice count, stop sequences |
| aabmass | 14 | 18 | Release prep, util-genai, upload hooks, version management |
| DylanRussell | 14 | 15 | Google GenAI/VertexAI token counting, streaming, semconv alignment |
| nagkumar91 | 8 | 17 | OpenAI Agents instrumentation (boilerplate through content capture) |
| wrisa | 6 | 14 | Langchain instrumentation (from boilerplate through genai utils integration) |
| vasantteja | 6 | 9 | Anthropic instrumentation, OpenAI Responses streams, service tier fixes |
| lmolkova | 5 | 7 | GenAI utils metrics, semconv 1.37.0 chat history, OpenAI examples |
| keith-decker | 5 | 8 | GenAI utils structure (metrics, semconv attributes, inference type) |
| emdneto | 3 | 3 | Python 3.14 support, CI improvements |
| wikaaaaa | 3 | 5 | Tool definitions, semconv alignment, JSON Schema compliance |

## Key Observations

### Volume and Velocity
- **OpenLLMetry generates more than 2× the PR volume** (384 vs 173), which is expected given it covers 20+ provider instrumentations and the full Traceloop SDK, compared to the focused GenAI subset of otel-python-contrib.
- **OpenLLMetry has a slightly higher merge rate** (63% vs 58%) over the full year.
- **OpenLLMetry merges dramatically faster** — median 0.9 days vs 5.0 days. The average gap is even wider (9.1 days vs 18.9 days). This reflects the difference between a company-maintained project that can merge quickly vs. an open-source foundation project with formal review processes.
- **OTel Python Contrib shows a clear ramp-up**: activity was low from March–August 2025 (4–9 PRs/month created, 0–4 merged/month), then jumped significantly starting September 2025 (18–36 PRs/month created, 7–17 merged/month). This inflection point coincides with increased investment in GenAI instrumentation from multiple contributors.
- **OpenLLMetry had a burst of activity in July–August 2025** (53–65 PRs created, 51–53 merged), likely reflecting a concerted push on features or semconv alignment.

### Contributor Profiles
- **OpenLLMetry has far more one-off contributors** (63 single-PR authors vs 24). Many community bug fixes or features were submitted but not merged, indicating a wider but shallower contributor base.
- **OpenLLMetry's top contributors are predominantly Traceloop employees** (galkleinman, nirga, nina-kollman, dinmukhamedm), with community contributions (LuizDMM, minimAluminiumalism, elinacse) making up a meaningful secondary layer.
- **OTel Python Contrib contributors are more diverse by affiliation** — Google (DylanRussell, aabmass), community volunteers (xrmx, vasantteja, nagkumar91, wrisa, keith-decker), and Microsoft (lmolkova).
- **OTel Python Contrib has stronger repeat contributor engagement** — 15 repeat contributors out of 47 uniques (32%) vs 25 out of 96 (26%) for OpenLLMetry.

### Nature of Work
- **OpenLLMetry** work is heavily focused on:
  - Bug fixes across existing provider instrumentations (OpenAI, Anthropic, Langchain, Ollama, Watsonx, etc.)
  - Traceloop SDK features (evals, datasets, conversation decorators, experiments)
  - New instrumentations (Voyage AI, Agno, LlamaParse)
  - Dependency/semconv version bumps and CI modernization (uv + ruff migration)
  - OTel GenAI semconv compliance effort (LuizDMM's event emission PRs, max-deygin-traceloop's alignment series)

- **OTel Python Contrib (GenAI)** work is focused on:
  - Building new provider instrumentations from scratch (Anthropic, Google GenAI, Langchain, OpenAI Agents, Weaviate)
  - GenAI utility library (util-genai) for shared metrics, semconv attributes, and inference patterns
  - Streaming support and response wrapper infrastructure
  - Semantic conventions alignment (1.37.0 chat history, tool definitions)
  - OpenAI Responses API support
  - Release engineering, CI improvements, and Python 3.14 readiness

### Activity Trends
- **OpenLLMetry's peak** was July–August 2025 with 118 PRs created and 104 merged across the two months — nearly one-third of the year's total output.
- **OTel Python Contrib's acceleration** starting September 2025 brought the project from ~5 PRs/month to ~20+ PRs/month. February–March 2026 averages 30 PRs/month, suggesting sustained growth.
- Both projects show increased activity in recent months, reflecting the growing importance of GenAI observability.

### Open PR Backlog
- **OpenLLMetry has 85 open PRs**, including ongoing semconv compliance migrations and community-submitted features awaiting review.
- **OTel Python Contrib has 34 open PRs**, including new instrumentation features (Anthropic agents, MistralAI) and GenAI utils enhancements.

### Convergence Signal
- Both projects are converging on OTel GenAI semantic conventions. OpenLLMetry has invested heavily in event emission and semconv alignment (LuizDMM contributed 23 merged PRs focused on this). OTel Python Contrib is building the reference implementation from the ground up with semconv compliance as a first-class concern.
- The util-genai shared library in OTel Python Contrib (driven by aabmass, keith-decker, lmolkova) represents an infrastructure investment that OpenLLMetry lacks — a common foundation for consistent behavior across instrumentations.
