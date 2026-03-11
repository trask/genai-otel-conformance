#!/usr/bin/env bash
# run_test.sh — Run a single conformance test against weaver live-check.
#
# Usage:
#   ./run_test.sh <test-name>
#
# Test name format: {lang}-{lib}-{ecosystem}
#   e.g. python-openai-otelcontrib, js-openai-openllmetry,
#        java-langchain4j-otelcontrib, dotnet-extensions-ai-native
#
# Requires:
#   - weaver on PATH
#   - python (for mock server), plus language-specific toolchain

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TEST_NAME="${1:?Usage: run_test.sh <test-name>}"
shift 1

# ── Auto-discover test command from test name ───────────────────────

discover_test_cmd() {
    local name="$1"

    # Parse: lang-lib-ecosystem (lib may contain hyphens, ecosystem is the last segment)
    local lang eco lib
    lang="${name%%-*}"
    local rest="${name#*-}"

    case "$lang" in
        python)
            eco="${rest##*-}"
            lib="${rest%-*}"
            local test_file="tests/python/${lib}/test_${eco}.py"
            if [[ -f "$test_file" ]]; then
                echo "python $test_file"
                return 0
            fi
            ;;
        js)
            eco="${rest##*-}"
            lib="${rest%-*}"
            local test_dir="tests/js/${lib}"
            local script="test:${eco}"
            if [[ -d "$test_dir" ]]; then
                echo "cd $test_dir && npm install --silent && npm run $script"
                return 0
            fi
            ;;
        java)
            eco="${rest##*-}"
            lib="${rest%-*}"
            local test_dir="tests/java/${lib}"
            if [[ -d "$test_dir" ]]; then
                if grep -q 'spring-boot' "$test_dir/build.gradle.kts" 2>/dev/null; then
                    echo "cd $test_dir && ./gradlew bootRun"
                else
                    echo "cd $test_dir && ./gradlew run"
                fi
                return 0
            fi
            ;;
        dotnet)
            eco="${rest##*-}"
            lib="${rest%-*}"
            local test_dir="tests/dotnet/${lib}"
            if [[ -d "$test_dir" ]]; then
                echo "cd $test_dir && dotnet run"
                return 0
            fi
            ;;
    esac

    return 1
}

TEST_CMD=$(discover_test_cmd "$TEST_NAME") || {
    echo "ERROR: Could not auto-discover test command for '$TEST_NAME'" >&2
    echo "Available tests:" >&2
    for f in tests/python/*/test_*.py; do
        [[ -f "$f" ]] || continue
        local_lib=$(basename "$(dirname "$f")")
        local_eco=$(basename "$f" .py | sed 's/^test_//')
        echo "  python-${local_lib}-${local_eco}" >&2
    done
    for f in tests/js/*/test_*.ts; do
        [[ -f "$f" ]] || continue
        local_lib=$(basename "$(dirname "$f")")
        local_eco=$(basename "$f" .ts | sed 's/^test_//')
        echo "  js-${local_lib}-${local_eco}" >&2
    done
    for f in tests/java/*/build.gradle.kts; do
        [[ -f "$f" ]] || continue
        local_lib=$(basename "$(dirname "$f")")
        if grep -q 'spring-boot' "$f" 2>/dev/null; then
            echo "  java-${local_lib}-native" >&2
        else
            echo "  java-${local_lib}-otelcontrib" >&2
        fi
    done
    for f in tests/dotnet/*/*.csproj; do
        [[ -f "$f" ]] || continue
        local_lib=$(basename "$(dirname "$f")")
        echo "  dotnet-${local_lib}-native" >&2
    done
    exit 1
}
echo "Test command: $TEST_CMD"

RESULTS_DIR="${RESULTS_DIR:-results}"
WEAVER_PORT="${WEAVER_PORT:-4317}"
ADMIN_PORT="${ADMIN_PORT:-4320}"
INACTIVITY_TIMEOUT="${INACTIVITY_TIMEOUT:-120}"
MOCK_PORT="${MOCK_PORT:-8080}"

# ── Default MOCK_LLM_URL ────────────────────────────────────────────

export MOCK_LLM_URL="${MOCK_LLM_URL:-http://127.0.0.1:$MOCK_PORT}"

# ── Auto-start mock server if not already running ───────────────────

MOCK_PID=""

cleanup() {
    if [[ -n "$MOCK_PID" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
        echo "Stopping mock server (PID $MOCK_PID)..."
        kill "$MOCK_PID" 2>/dev/null || true
        wait "$MOCK_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if ! curl -s "http://127.0.0.1:$MOCK_PORT/health" > /dev/null 2>&1; then
    echo "=== Starting mock server on port $MOCK_PORT ==="
    pip install -q -e "$SCRIPT_DIR/mock-server/" 2>/dev/null || true
    python "$SCRIPT_DIR/mock-server/mock_server/server.py" &
    MOCK_PID=$!
    for i in $(seq 1 30); do
        if curl -s "http://127.0.0.1:$MOCK_PORT/health" > /dev/null 2>&1; then
            echo "Mock server ready after ${i}s"
            break
        fi
        if ! kill -0 "$MOCK_PID" 2>/dev/null; then
            echo "ERROR: Mock server failed to start" >&2
            exit 1
        fi
        sleep 1
    done
else
    echo "Mock server already running on port $MOCK_PORT"
fi

mkdir -p "$RESULTS_DIR/$TEST_NAME"

# Wait for ports to be fully released (including TIME_WAIT)
echo "=== Waiting for ports to be available ==="
for i in $(seq 1 30); do
    grpc_busy=$(netstat -ano 2>/dev/null | grep -c ":${WEAVER_PORT} " || true)
    admin_busy=$(netstat -ano 2>/dev/null | grep -c ":${ADMIN_PORT} " || true)
    if [[ "$grpc_busy" -eq 0 && "$admin_busy" -eq 0 ]]; then
        echo "Ports $WEAVER_PORT/$ADMIN_PORT free after ${i}s"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "WARNING: Ports still busy after 30s (gRPC=$grpc_busy, admin=$admin_busy), proceeding anyway"
    fi
    sleep 2
done

echo "=== Starting weaver live-check for: $TEST_NAME ==="
weaver registry live-check \
    --format json \
    --output "$RESULTS_DIR/$TEST_NAME" \
    --otlp-grpc-port "$WEAVER_PORT" \
    --admin-port "$ADMIN_PORT" \
    --inactivity-timeout "$INACTIVITY_TIMEOUT" \
    "$@" &
WEAVER_PID=$!

# Wait for weaver admin port to be ready (confirms gRPC is also ready)
echo "Waiting for weaver to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$ADMIN_PORT/status" > /dev/null 2>&1; then
        echo "Weaver ready after ${i}s"
        break
    fi
    # Check if weaver died
    if ! kill -0 "$WEAVER_PID" 2>/dev/null; then
        echo "ERROR: Weaver process died during startup"
        wait "$WEAVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo "=== Running test: $TEST_CMD ==="
WEAVER_EXIT=0

# Export the weaver port so test scripts know where to send telemetry
export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:$WEAVER_PORT"

# Run the test; capture exit code but don't fail the script yet
eval "$TEST_CMD" || true

# Give weaver a moment to process final telemetry
sleep 2

# Signal weaver to stop
if kill -0 "$WEAVER_PID" 2>/dev/null; then
    # Use the /stop endpoint for a clean shutdown
    curl -s -X POST "http://localhost:$ADMIN_PORT/stop" > /dev/null 2>&1 || \
        kill -HUP "$WEAVER_PID" 2>/dev/null || true
fi

wait "$WEAVER_PID" || WEAVER_EXIT=$?

echo "=== Weaver exit code: $WEAVER_EXIT ==="
echo "=== Results in: $RESULTS_DIR/$TEST_NAME ==="
