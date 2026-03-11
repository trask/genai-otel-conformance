#!/usr/bin/env bash
# run_all.sh — Run all (or selected) conformance tests locally and generate the dashboard.
#
# Usage:
#   ./run_all.sh                        # Run all tests
#   ./run_all.sh openai langchain       # Run only those libraries
#   ./run_all.sh --list                 # List available tests
#   ./run_all.sh --dashboard-only       # Regenerate dashboard from existing results
#
# Prerequisites:
#   - weaver on PATH
#   - uv (pip install uv) OR pip install -e mock-server/
#   - For JS tests: node/npm
#   - For Java tests: JDK 17+ and gradlew in test dirs
#   - For .NET tests: dotnet CLI

set -uo pipefail

RESULTS_DIR="${RESULTS_DIR:-results}"
MOCK_PORT="${MOCK_PORT:-8080}"
MOCK_PID=""
VENV_BASE="/tmp/otel-venvs"

# ── Helpers ──────────────────────────────────────────────────────────

cleanup() {
    if [[ -n "$MOCK_PID" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
        echo "Stopping mock server (PID $MOCK_PID)..."
        kill "$MOCK_PID" 2>/dev/null || true
        wait "$MOCK_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_mock_server() {
    echo "=== Starting mock server on port $MOCK_PORT ==="
    python "$(dirname "$0")/mock-server/mock_server/server.py" &
    MOCK_PID=$!
    # Wait for it to be ready
    for i in $(seq 1 30); do
        if curl -s "http://127.0.0.1:$MOCK_PORT/health" > /dev/null 2>&1; then
            echo "Mock server ready."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Mock server failed to start" >&2
    return 1
}

# ── Discover tests ───────────────────────────────────────────────────

declare -a ALL_TESTS=()

discover_python_tests() {
    for test_file in tests/python/*/test_*.py; do
        [[ -f "$test_file" ]] || continue
        local dir lib eco name reqs
        dir=$(dirname "$test_file")
        lib=$(basename "$dir")
        eco=$(basename "$test_file" .py | sed 's/^test_//')
        name="python-${lib}-${eco}"
        reqs="${dir}/requirements-${eco}.txt"
        ALL_TESTS+=("python|${name}|${test_file}|${reqs}")
    done
}

discover_js_tests() {
    for test_file in tests/js/*/test_*.ts; do
        [[ -f "$test_file" ]] || continue
        local dir lib eco name
        dir=$(dirname "$test_file")
        lib=$(basename "$dir")
        eco=$(basename "$test_file" .ts | sed 's/^test_//')
        name="js-${lib}-${eco}"
        ALL_TESTS+=("js|${name}|${test_file}|${dir}")
    done
}

discover_java_tests() {
    for build_file in tests/java/*/build.gradle.kts; do
        [[ -f "$build_file" ]] || continue
        local dir lib
        dir=$(dirname "$build_file")
        lib=$(basename "$dir")
        # Determine ecosystem from build.gradle.kts — check for spring-boot plugin or otel agent
        if grep -q 'spring-boot' "$build_file" 2>/dev/null; then
            local eco="native"
        else
            local eco="otelcontrib"
        fi
        local name="java-${lib}-${eco}"
        ALL_TESTS+=("java|${name}|${dir}|${dir}")
    done
}

discover_dotnet_tests() {
    for proj_file in tests/dotnet/*/*.csproj; do
        [[ -f "$proj_file" ]] || continue
        local dir lib name
        dir=$(dirname "$proj_file")
        lib=$(basename "$dir")
        name="dotnet-${lib}-native"
        ALL_TESTS+=("dotnet|${name}|${dir}|${dir}")
    done
}

list_tests() {
    discover_python_tests
    discover_js_tests
    discover_java_tests
    discover_dotnet_tests
    echo "Available tests:"
    printf "  %-40s %-8s %s\n" "NAME" "LANG" "FILE"
    for entry in "${ALL_TESTS[@]}"; do
        IFS='|' read -r lang name file reqs <<< "$entry"
        printf "  %-40s %-8s %s\n" "$name" "$lang" "$file"
    done
}

# ── Run a single test ────────────────────────────────────────────────

run_one_test() {
    local lang="$1" name="$2" file="$3" reqs="$4"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  $name ($lang)"
    echo "╚══════════════════════════════════════════════════════════╝"

    export MOCK_LLM_URL="http://127.0.0.1:$MOCK_PORT"

    # Each test gets unique ports to avoid TIME_WAIT conflicts
    export WEAVER_PORT=$((WEAVER_PORT_BASE + TEST_INDEX * 2))
    export ADMIN_PORT=$((WEAVER_PORT_BASE + TEST_INDEX * 2 + 1))
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:$WEAVER_PORT"
    TEST_INDEX=$((TEST_INDEX + 1))

    local test_cmd=""

    case "$lang" in
        python)
            local venv_dir="$VENV_BASE/$name"
            if [[ ! -d "$venv_dir" ]]; then
                echo "Creating venv: $venv_dir"
                "$SYSTEM_PYTHON" -m venv "$venv_dir"
                # Install shared otel_setup package
                "$SYSTEM_PYTHON" -m uv pip install -q -e tests/python --python "$venv_dir" 2>/dev/null || true
                if [[ -f "$reqs" ]]; then
                    echo "Installing deps: $reqs"
                    "$SYSTEM_PYTHON" -m uv pip install -q -r "$reqs" --python "$venv_dir" 2>/dev/null || {
                        echo "WARNING: Failed to install $reqs — skipping test"
                        rm -rf "$venv_dir"
                        return 0
                    }
                fi
            else
                echo "Reusing venv: $venv_dir"
            fi
            # Activate the per-test venv
            source "$venv_dir/Scripts/activate" 2>/dev/null || source "$venv_dir/bin/activate"
            # Record installed package versions for reproducibility
            pip freeze > "$RESULTS_DIR/$name/frozen-requirements.txt" 2>/dev/null || true
            # Save dependency versions as JSON for dashboard
            pip list --format json 2>/dev/null | python3 -c "import sys,json;json.dump({p['name']:p['version'] for p in json.load(sys.stdin)},sys.stdout,indent=2)" > "$RESULTS_DIR/$name/versions.json" 2>/dev/null || true
            test_cmd="python $file"
            ;;
        js)
            local dir="$reqs"  # for JS, reqs field holds the directory
            echo "Installing npm deps in $dir"
            (cd "$dir" && npm install --silent 2>/dev/null) || {
                echo "WARNING: npm install failed in $dir — skipping test"
                return 0
            }
            # Save dependency versions as JSON for dashboard
            mkdir -p "$RESULTS_DIR/$name"
            python3 -c "
import json,os,sys
d=sys.argv[1]
pkg=json.load(open(os.path.join(d,'package.json')))
v={}
for n,s in pkg.get('dependencies',{}).items():
    m=os.path.join(d,'node_modules',n,'package.json')
    try: v[n]=json.load(open(m)).get('version',s)
    except: v[n]=s
json.dump(v,sys.stdout,indent=2)
" "$dir" > "$RESULTS_DIR/$name/versions.json" 2>/dev/null || true
            test_cmd="(cd $dir && npx tsx $(basename "$file"))"
            ;;
        java)
            local dir="$reqs"  # for Java, reqs field holds the directory
            echo "Building and running Java test in $dir"
            # Save dependency versions as JSON for dashboard
            mkdir -p "$RESULTS_DIR/$name"
            python3 -c "
import re,json,sys
v={}
with open(sys.argv[1]) as f:
    for line in f:
        m=re.search(r'implementation\(\"(.*?):(.*?):(.*?)\"\)',line)
        if m: v[m.group(1)+':'+m.group(2)]=m.group(3)
json.dump(v,sys.stdout,indent=2)
" "$dir/build.gradle.kts" > "$RESULTS_DIR/$name/versions.json" 2>/dev/null || true
            if grep -q 'spring-boot' "$dir/build.gradle.kts" 2>/dev/null; then
                test_cmd="(cd $dir && ./gradlew bootRun --quiet)"
            else
                test_cmd="(cd $dir && ./gradlew run --quiet)"
            fi
            ;;
        dotnet)
            local dir="$reqs"  # for .NET, reqs field holds the directory
            echo "Running .NET test in $dir"
            # Save dependency versions as JSON for dashboard
            mkdir -p "$RESULTS_DIR/$name"
            python3 -c "
import xml.etree.ElementTree as ET,json,sys,glob
v={}
for f in glob.glob(sys.argv[1]+'/*.csproj'):
    for r in ET.parse(f).findall('.//PackageReference'):
        n=r.get('Include','')
        if n: v[n]=r.get('Version','')
json.dump(v,sys.stdout,indent=2)
" "$dir" > "$RESULTS_DIR/$name/versions.json" 2>/dev/null || true
            test_cmd="(cd $dir && dotnet run)"
            ;;
        *)
            echo "WARNING: Unknown language $lang — skipping"
            return 0
            ;;
    esac

    local exit_code=0
    ./run_test.sh "$name" || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "  ✓ PASS"
    else
        echo "  ✗ FAIL (exit code $exit_code)"
    fi

    return $exit_code
}

# ── Main ─────────────────────────────────────────────────────────────

main() {
    local dashboard_only=false
    local -a filters=()

    # Parse args
    for arg in "$@"; do
        case "$arg" in
            --list)
                list_tests
                exit 0
                ;;
            --dashboard-only)
                dashboard_only=true
                ;;
            --help|-h)
                head -12 "$0" | tail -10
                exit 0
                ;;
            *)
                filters+=("$arg")
                ;;
        esac
    done

    if [[ "$dashboard_only" == true ]]; then
        echo "=== Generating dashboard from existing results ==="
        python generate_dashboard.py --results-dir "$RESULTS_DIR" --output-dir docs
        echo "Open docs/index.html in your browser."
        exit 0
    fi

    # Check prerequisites
    if ! command -v weaver &>/dev/null; then
        echo "ERROR: weaver not found on PATH. Install from:" >&2
        echo "  https://github.com/open-telemetry/weaver/releases" >&2
        exit 1
    fi

    # Create or reuse Python venv
    SYSTEM_PYTHON=$(command -v python)

    discover_python_tests
    discover_js_tests
    discover_java_tests
    discover_dotnet_tests

    # Filter tests if library names specified
    if [[ ${#filters[@]} -gt 0 ]]; then
        local -a filtered=()
        for entry in "${ALL_TESTS[@]}"; do
            IFS='|' read -r lang name file reqs <<< "$entry"
            for f in "${filters[@]}"; do
                if [[ "$name" == *"$f"* ]]; then
                    filtered+=("$entry")
                    break
                fi
            done
        done
        ALL_TESTS=("${filtered[@]}")
    fi

    echo "=== Running ${#ALL_TESTS[@]} conformance tests ==="

    start_mock_server

    # Port base for weaver instances — each test uses WEAVER_PORT_BASE + i*2 (gRPC) and +i*2+1 (admin)
    WEAVER_PORT_BASE=14317
    TEST_INDEX=0

    local passed=0 failed=0 skipped=0
    for entry in "${ALL_TESTS[@]}"; do
        IFS='|' read -r lang name file reqs <<< "$entry"
        if run_one_test "$lang" "$name" "$file" "$reqs"; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done

    echo ""
    echo "=== Results: $passed passed, $failed failed (${#ALL_TESTS[@]} total) ==="
    echo "=== Generating dashboard ==="
    python generate_dashboard.py --results-dir "$RESULTS_DIR" --output-dir docs
    echo ""
    echo "Done! Open docs/index.html in your browser."
}

main "$@"
