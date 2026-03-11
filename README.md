# GenAI OpenTelemetry Conformance Tests

Automated conformance testing of GenAI framework instrumentations against the [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

## How It Works

Each test app sends requests to a **mock LLM server** (no API keys needed), with OpenTelemetry instrumentation exporting telemetry via OTLP to [**Weaver `registry live-check`**](https://github.com/open-telemetry/weaver). Weaver validates every span, metric, and log against the official semconv registry and reports violations, improvements, and coverage statistics.

## Quick Start

Prerequisites: [Weaver CLI](https://github.com/open-telemetry/weaver), Python 3.12+, and optionally Node.js 20+ / .NET 8+ / Java 21+ for non-Python tests.

```bash
# Run a test (auto-starts mock server, auto-discovers test command)
./run_test.sh python-openai-otelcontrib
```

Results appear in `results/<test-name>/`. Push to `main` or open a PR to run all tests via CI.

## Dashboard

The conformance dashboard is auto-generated after CI and deployed to GitHub Pages. Generate locally:

```bash
python generate_dashboard.py --results-dir results --output-dir docs
```

## Adding a New Test

1. Create `tests/<lang>/<library>/test_<ecosystem>.py` following the existing pattern
2. Create `tests/<lang>/<library>/requirements-<ecosystem>.txt`
3. Add a matrix entry in `.github/workflows/ci.yml`

## License

Apache-2.0
