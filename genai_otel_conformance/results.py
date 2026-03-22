from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

LANGUAGE_DISPLAY_NAMES = {"python": "Python", "java": "Java", "js": "JS", "dotnet": "C#"}


def _load_ecosystems() -> tuple[
    dict[str, str],
    dict[tuple[str, str], str],
]:
    """Load ecosystem definitions from tests/ecosystems.json."""
    eco_file = TESTS_DIR / "ecosystems.json"
    if not eco_file.is_file():
        return {}, {}
    data = json.loads(eco_file.read_text(encoding="utf-8"))
    display: dict[str, str] = {}
    repos: dict[tuple[str, str], str] = {}
    for eco, info in data.items():
        display[eco] = info.get("display_name", eco)
        for lang_slug, repo in info.get("repos", {}).items():
            lang_display = LANGUAGE_DISPLAY_NAMES.get(lang_slug, lang_slug)
            repos[(eco, lang_display)] = repo
    return display, repos


ECOSYSTEM_DISPLAY, ECOSYSTEM_REPOS = _load_ecosystems()


@lru_cache(maxsize=1)
def _discover_library_metadata() -> tuple[
    dict[str, str],
    dict[tuple[str, str], str],
]:
    """Scan metadata.json files for library display names and native repos."""
    names: dict[str, str] = {}
    repos: dict[tuple[str, str], str] = {}
    if not TESTS_DIR.is_dir():
        return names, repos
    for lang_dir in sorted(TESTS_DIR.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name not in LANGUAGE_DISPLAY_NAMES:
            continue
        for lib_dir in sorted(lang_dir.iterdir()):
            if not lib_dir.is_dir():
                continue
            slug = lib_dir.name
            meta = lib_dir / "metadata.json"
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if slug not in names and "display_name" in data:
                names[slug] = data["display_name"]
            if "repo" in data:
                repos[(lang_dir.name, slug)] = data["repo"]
    return names, repos


LIBRARY_DISPLAY_NAMES, NATIVE_REPOS = _discover_library_metadata()


def library_display_name(slug: str) -> str:
    """Return the human-readable display name for a library slug."""
    return LIBRARY_DISPLAY_NAMES.get(slug, slug)


class TestName(NamedTuple):
    language: str
    library: str
    ecosystem: str


@dataclass
class TestResult:
    language: str
    library: str
    ecosystem: str
    statistics: dict | None
    violation_count: int
    violation_messages: list[str]
    entity_counts: dict[str, int]
    seen_attrs: dict[str, int]
    seen_non_registry_attrs: dict[str, int]
    seen_events: dict[str, int]
    seen_metrics: dict[str, int]
    has_data: bool
    detected_span_types: set[str] = field(default_factory=set)
    per_type_attrs: dict[str, set[str]] = field(default_factory=dict)
    detected_events: dict[str, int] = field(default_factory=dict)
    detected_metrics: dict[str, int] = field(default_factory=dict)


def split_test_name(name: str) -> tuple[str, str, str]:
    """Parse a test name into language/library/ecosystem slugs."""
    try:
        lang, rest = name.split("-", 1)
        lib, eco = rest.rsplit("-", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid test name: {name}") from exc

    if lang not in LANGUAGE_DISPLAY_NAMES or not lib or not eco:
        raise ValueError(f"Invalid test name: {name}")

    return lang, lib, eco


def parse_test_name(test_name: str) -> TestName:
    """Parse a supported test name into display values."""
    lang, library, ecosystem = split_test_name(test_name)
    return TestName(LANGUAGE_DISPLAY_NAMES[lang], library, ecosystem)


def try_parse_json(content: str) -> list[dict]:
    """Parse JSON content, handling a single object, array, or JSONL."""
    objects: list[dict] = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        objects.extend(data)
        return objects
    if isinstance(data, dict):
        objects.append(data)
        return objects

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return objects


_COMMON_REQUIRED = ["gen_ai.operation.name"]
_PROVIDER_REQUIRED = ["gen_ai.provider.name", "gen_ai.system"]
_COMMON_COND_REQUIRED = ["error.type"]
_CLIENT_COND_REQUIRED = ["gen_ai.request.model", "server.port"]
_CLIENT_RECOMMENDED = ["server.address"]
_INFERENCE_COND_REQUIRED = [
    "gen_ai.conversation.id",
    "gen_ai.output.type",
    "gen_ai.request.choice.count",
    "gen_ai.request.seed",
]
_INFERENCE_RECOMMENDED = [
    "gen_ai.request.frequency_penalty",
    "gen_ai.request.max_tokens",
    "gen_ai.request.presence_penalty",
    "gen_ai.request.stop_sequences",
    "gen_ai.request.temperature",
    "gen_ai.request.top_p",
    "gen_ai.response.finish_reasons",
    "gen_ai.response.id",
    "gen_ai.response.model",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
]

SPAN_TYPE_SPECS = {
    "inference": {
        "label": "Inference",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.response.finish_reasons", "gen_ai.response.id",
            "gen_ai.usage.output_tokens", "gen_ai.request.max_tokens",
            "gen_ai.request.temperature", "gen_ai.output.type",
            "gen_ai.usage.input_tokens",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INFERENCE_COND_REQUIRED,
        "recommended": _INFERENCE_RECOMMENDED + ["gen_ai.request.top_k"] + _CLIENT_RECOMMENDED,
    },
    "embeddings": {
        "label": "Embeddings",
        "expected_kind": "client",
        "discriminator_attrs": {
            "gen_ai.embeddings.dimension.count", "gen_ai.request.encoding_formats",
        },
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED,
        "recommended": [
            "gen_ai.embeddings.dimension.count",
            "gen_ai.request.encoding_formats",
            "gen_ai.response.model",
            "gen_ai.usage.input_tokens",
        ] + _CLIENT_RECOMMENDED,
    },
    "retrieval": {
        "label": "Retrieval",
        "expected_kind": "client",
        "discriminator_attrs": {"gen_ai.data_source.id"},
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + [
            "gen_ai.data_source.id",
            "gen_ai.provider.name",
            "gen_ai.system",
        ] + _CLIENT_COND_REQUIRED,
        "recommended": ["gen_ai.request.top_k"] + _CLIENT_RECOMMENDED,
    },
    "execute_tool": {
        "label": "Execute Tool",
        "expected_kind": "internal",
        "discriminator_attrs": {
            "gen_ai.tool.call.id", "gen_ai.tool.name", "gen_ai.tool.type",
        },
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED,
        "recommended": [
            "gen_ai.tool.call.id",
            "gen_ai.tool.description",
            "gen_ai.tool.name",
            "gen_ai.tool.type",
        ],
    },
    "create_agent": {
        "label": "Create Agent",
        "expected_kind": "client",
        "discriminator_attrs": {"gen_ai.agent.id", "gen_ai.agent.name"},
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.agent.version",
        ],
        "recommended": _CLIENT_RECOMMENDED,
    },
    "invoke_agent": {
        "label": "Invoke Agent",
        "expected_kind": "client",
        "discriminator_attrs": {"gen_ai.agent.id", "gen_ai.agent.name"},
        "required": _COMMON_REQUIRED + _PROVIDER_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + _CLIENT_COND_REQUIRED + _INFERENCE_COND_REQUIRED + [
            "gen_ai.agent.description",
            "gen_ai.agent.id",
            "gen_ai.agent.name",
            "gen_ai.agent.version",
            "gen_ai.data_source.id",
        ],
        "recommended": _INFERENCE_RECOMMENDED + _CLIENT_RECOMMENDED,
    },
    "invoke_workflow": {
        "label": "Invoke Workflow",
        "expected_kind": "internal",
        "discriminator_attrs": {"gen_ai.workflow.name"},
        "required": _COMMON_REQUIRED,
        "conditionally_required": _COMMON_COND_REQUIRED + ["gen_ai.workflow.name"],
        "recommended": [],
    },
}

SPAN_TYPE_ORDER = [
    "create_agent",
    "invoke_agent",
    "invoke_workflow",
    "inference",
    "embeddings",
    "retrieval",
    "execute_tool",
]

GENAI_EVENT_TYPES = [
    "gen_ai.system.message",
    "gen_ai.user.message",
    "gen_ai.assistant.message",
    "gen_ai.tool.message",
    "gen_ai.choice",
]

GENAI_METRIC_TYPES = [
    "gen_ai.client.token.usage",
    "gen_ai.client.operation.duration",
]


@lru_cache(maxsize=None)
def _load_test_metadata(lang: str, library: str) -> dict:
    """Load metadata.json for a test directory."""
    meta_file = TESTS_DIR / lang / library / "metadata.json"
    if not meta_file.is_file():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _version_package_from_metadata(lang: str, library: str, ecosystem: str) -> str:
    metadata = _load_test_metadata(lang, library)
    version_packages = metadata.get("version_packages", {})
    if not isinstance(version_packages, dict):
        return ""
    package_name = version_packages.get(ecosystem, "")
    return package_name if isinstance(package_name, str) else ""


def _read_python_dependency_versions(test_dir: Path, ecosystem: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    req_file = test_dir / f"requirements-{ecosystem}.txt"
    if not req_file.exists():
        return versions
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "==" not in line:
            continue
        pkg, ver = line.split("==", 1)
        versions[pkg.strip()] = ver.strip()
    return versions


def _read_js_dependency_versions(test_dir: Path) -> dict[str, str]:
    pkg_file = test_dir / "package.json"
    if not pkg_file.exists():
        return {}
    try:
        data = json.loads(pkg_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data.get("dependencies", {}))


def _read_java_dependency_versions(test_dir: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    gradle_file = test_dir / "build.gradle.kts"
    if not gradle_file.exists():
        return versions
    content = gradle_file.read_text(encoding="utf-8")
    for match in re.finditer(r'implementation\("([^"]+)"\)', content):
        coord = match.group(1)
        parts = coord.rsplit(":", 1)
        if len(parts) == 2:
            versions[parts[0]] = parts[1]
    return versions


def _read_dotnet_dependency_versions(test_dir: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for csproj in test_dir.glob("*.csproj"):
        content = csproj.read_text(encoding="utf-8")
        for match in re.finditer(
            r'PackageReference\s+Include="([^"]+)"\s+Version="([^"]+)"',
            content,
        ):
            versions[match.group(1)] = match.group(2)
    return versions


def _read_deps_from_test_dir(lang: str, library: str, ecosystem: str) -> dict[str, str]:
    test_dir = TESTS_DIR / lang / library
    if lang == "python":
        return _read_python_dependency_versions(test_dir, ecosystem)
    if lang == "js":
        return _read_js_dependency_versions(test_dir)
    if lang == "java":
        return _read_java_dependency_versions(test_dir)
    if lang == "dotnet":
        return _read_dotnet_dependency_versions(test_dir)
    return {}


def extract_version_from_deps(lang: str, library: str, ecosystem: str) -> str:
    """Extract the display version from checked-in dependency files."""
    versions = _read_deps_from_test_dir(lang, library, ecosystem)
    package_name = _version_package_from_metadata(lang, library, ecosystem)
    if not package_name:
        return ""
    return versions.get(package_name, "")


def _classify_span(span_name: str, span_attrs: dict[str, object]) -> set[str]:
    """Classify a span into span types using heuristics on individual span data."""
    types: set[str] = set()
    name_lower = span_name.lower()
    op_name = str(span_attrs.get("gen_ai.operation.name", "")).lower()
    oi_kind = str(span_attrs.get("openinference.span.kind", "")).upper()
    llm_type = str(span_attrs.get("llm.request.type", "")).lower()

    if "embed" in name_lower:
        types.add("embeddings")
    elif span_attrs.get("embedding.model_name"):
        types.add("embeddings")
    elif oi_kind == "EMBEDDING":
        types.add("embeddings")
    elif llm_type in ("embedding", "embeddings"):
        types.add("embeddings")
    elif op_name in ("embedding", "embeddings"):
        types.add("embeddings")

    if op_name == "chat":
        types.add("inference")
    elif oi_kind == "LLM":
        types.add("inference")
    elif llm_type in ("chat", "completion"):
        types.add("inference")
    elif op_name == "generate_content":
        types.add("inference")
    elif span_attrs.get("gen_ai.usage.output_tokens") is not None \
            and span_attrs.get("gen_ai.response.finish_reasons") is not None:
        types.add("inference")
    elif span_attrs.get("llm.response.model") is not None \
            and span_attrs.get("llm.usage.completion_tokens") is not None:
        types.add("inference")

    if op_name == "create_agent":
        types.add("create_agent")

    if oi_kind == "AGENT":
        types.add("invoke_agent")
    elif op_name == "invoke_agent":
        types.add("invoke_agent")
    elif span_attrs.get("gen_ai.agent.name") or span_attrs.get("gen_ai.agent.id"):
        if op_name != "create_agent":
            types.add("invoke_agent")
    elif span_attrs.get("crewai.agent.id") or span_attrs.get("crewai.agent.role"):
        types.add("invoke_agent")
    # AWS SDK instrumentation: BedrockAgentRuntime.InvokeAgent
    elif str(span_attrs.get("rpc.service", "")).lower() == "bedrockagentruntime" \
            and str(span_attrs.get("rpc.method", "")).lower() == "invokeagent":
        types.add("invoke_agent")
    # Azure AI Foundry Agent: azure-core-tracing-opentelemetry spans
    # e.g. "AgentsClient.CreateAndProcessRun"
    elif "agentsclient" in name_lower and ("run" in name_lower or "process" in name_lower):
        types.add("invoke_agent")
    # OpenAI Assistants: threads.runs spans from opentelemetry-instrumentation-openai-v2
    elif "threads" in name_lower and "run" in name_lower \
            and "thread.run" not in name_lower:
        types.add("invoke_agent")

    if op_name == "execute_tool":
        types.add("execute_tool")
    elif oi_kind == "TOOL":
        types.add("execute_tool")
    elif span_attrs.get("gen_ai.tool.name") or span_attrs.get("gen_ai.tool.call.id"):
        types.add("execute_tool")

    if op_name == "invoke_workflow":
        types.add("invoke_workflow")
    elif span_attrs.get("traceloop.workflow.name"):
        types.add("invoke_workflow")
    elif name_lower == "crewai.workflow":
        types.add("invoke_workflow")
    elif span_attrs.get("crewai.crew.id"):
        types.add("invoke_workflow")

    if op_name == "retrieval":
        types.add("retrieval")
    elif oi_kind == "RETRIEVER":
        types.add("retrieval")
    elif span_attrs.get("gen_ai.data_source.id"):
        types.add("retrieval")

    return types


def _extract_span_types_from_samples(
    all_objects: list[dict],
) -> tuple[set[str], dict[str, set[str]]]:
    """Scan sample spans, classify them, and track per-type attrs."""
    span_types: set[str] = set()
    per_type_attrs: dict[str, set[str]] = {}
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            span = sample.get("span")
            if not span:
                continue
            attrs: dict[str, object] = {}
            for attr in span.get("attributes", []):
                attrs[attr.get("name", "")] = attr.get("value")
            classified = _classify_span(span.get("name", ""), attrs)
            span_types |= classified
            attr_names = set(attrs.keys())
            for span_type in classified:
                if span_type not in per_type_attrs:
                    per_type_attrs[span_type] = set()
                per_type_attrs[span_type] |= attr_names
    return span_types, per_type_attrs


def _extract_events_from_samples(all_objects: list[dict]) -> dict[str, int]:
    events: dict[str, int] = {}
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            log = sample.get("log")
            if not log:
                continue
            event_name = log.get("event_name", "")
            if event_name.startswith("gen_ai."):
                events[event_name] = events.get(event_name, 0) + 1
    return events


def _extract_metrics_from_samples(all_objects: list[dict]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        for sample in obj.get("samples", []):
            metric = sample.get("metric")
            if not metric:
                continue
            metric_name = metric.get("name", "")
            if metric_name.startswith("gen_ai."):
                metrics[metric_name] = metrics.get(metric_name, 0) + 1
    return metrics


def _non_zero_counts(statistics: dict | None, key: str) -> dict[str, int]:
    if not statistics:
        return {}
    return {
        name: count
        for name, count in statistics.get(key, {}).items()
        if count > 0
    }


def parse_result_dir(result_dir: Path, test_name: str) -> TestResult | None:
    """Parse a single test's Weaver output directory into a TestResult."""
    if not result_dir.is_dir():
        return None

    all_objects: list[dict] = []
    for json_file in sorted(result_dir.glob("**/*.json")):
        try:
            all_objects.extend(try_parse_json(json_file.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"Warning: Could not parse {json_file}: {exc}", file=sys.stderr)

    statistics = None
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        if "statistics" in obj and isinstance(obj["statistics"], dict):
            statistics = obj["statistics"]
        elif "registry_coverage" in obj or "advice_level_counts" in obj:
            statistics = obj

    seen_attrs = _non_zero_counts(statistics, "seen_registry_attributes")
    seen_non_registry_attrs = _non_zero_counts(statistics, "seen_non_registry_attributes")
    seen_events = {
        **_non_zero_counts(statistics, "seen_registry_events"),
        **_non_zero_counts(statistics, "seen_non_registry_events"),
    }
    seen_metrics = {
        **_non_zero_counts(statistics, "seen_registry_metrics"),
        **_non_zero_counts(statistics, "seen_non_registry_metrics"),
    }

    violation_count = 0
    if statistics:
        violation_count = statistics.get("advice_level_counts", {}).get("violation", 0)

    violation_messages: set[str] = set()
    if statistics:
        for msg in statistics.get("advice_message_counts", {}):
            if "not stable" not in msg.lower():
                violation_messages.add(msg)

    entity_counts = statistics.get("total_entities_by_type", {}) if statistics else {}

    try:
        language, library, ecosystem = parse_test_name(test_name)
    except ValueError:
        print(f"Warning: Could not parse test name: {test_name}", file=sys.stderr)
        return None

    has_data = bool(statistics and statistics.get("total_entities", 0) > 0)
    detected_span_types, per_type_attrs = _extract_span_types_from_samples(all_objects)
    detected_events = _extract_events_from_samples(all_objects)
    detected_metrics = _extract_metrics_from_samples(all_objects)

    if statistics:
        for ev_name, count in statistics.get("seen_non_registry_events", {}).items():
            if count > 0 and ev_name.startswith("gen_ai."):
                detected_events[ev_name] = max(detected_events.get(ev_name, 0), count)

        for metric_name, count in statistics.get("seen_non_registry_metrics", {}).items():
            if count > 0 and metric_name.startswith("gen_ai."):
                detected_metrics[metric_name] = max(detected_metrics.get(metric_name, 0), count)

    return TestResult(
        language=language,
        library=library,
        ecosystem=ecosystem,
        statistics=statistics,
        violation_count=violation_count,
        violation_messages=sorted(violation_messages),
        entity_counts=entity_counts,
        seen_attrs=seen_attrs,
        seen_non_registry_attrs=seen_non_registry_attrs,
        seen_events=seen_events,
        seen_metrics=seen_metrics,
        has_data=has_data,
        detected_span_types=detected_span_types,
        per_type_attrs=per_type_attrs,
        detected_events=detected_events,
        detected_metrics=detected_metrics,
    )