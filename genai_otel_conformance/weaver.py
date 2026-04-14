"""Weaver binary management: version pinning, download, and caching."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from genai_otel_conformance import REPO_ROOT

POLICIES_SOURCE_DIR = REPO_ROOT / "policies"

# Maps Rego constant name → schema filename in docs/gen-ai/ of the semconv repo.
_GENAI_SCHEMA_FILES: dict[str, str] = {
    "input_messages":      "gen-ai-input-messages.json",
    "output_messages":     "gen-ai-output-messages.json",
    "system_instructions": "gen-ai-system-instructions.json",
}

VERSIONS_FILE = REPO_ROOT / "versions.env"


def _load_version_pins() -> dict[str, str]:
    """Load shared external version pins from the repository root."""
    try:
        content = VERSIONS_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read version pins file: {VERSIONS_FILE}") from exc

    pins: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise RuntimeError(f"Invalid version pin line in {VERSIONS_FILE}: {raw_line!r}")
        pins[key.strip()] = value.strip().strip('"').strip("'")

    return pins


VERSION_PINS = _load_version_pins()
WEAVER_VERSION = VERSION_PINS["WEAVER_VERSION"]
SEMCONV_VERSION = VERSION_PINS["SEMCONV_VERSION"]


def _normalize_version(version: str) -> str:
    """Normalize optional leading-v version strings for comparisons."""
    return version[1:] if version.startswith("v") else version


def path_from_env(var_name: str, default_path: Path) -> Path:
    value = os.environ.get(var_name)
    if value:
        return Path(value)
    return default_path


def _weaver_binary_name() -> str:
    """Return the platform-specific Weaver binary name."""
    return "weaver.exe" if sys.platform == "win32" else "weaver"


def _weaver_asset_name() -> str:
    """Return the pinned Weaver release asset name for the current platform."""
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-pc-windows-msvc.zip"
    if sys.platform == "linux" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-unknown-linux-gnu.tar.xz"
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "weaver-aarch64-apple-darwin.tar.xz"
    if sys.platform == "darwin" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-apple-darwin.tar.xz"
    raise RuntimeError(
        f"Unsupported platform for managed Weaver install: {sys.platform} / {platform.machine()}"
    )


def _find_weaver_binary(search_root: Path) -> Path | None:
    """Return the first Weaver binary found under the given directory."""
    for path in search_root.rglob(_weaver_binary_name()):
        if path.is_file():
            return path
    return None


def _weaver_version(binary: str | Path) -> str:
    """Return the normalized version reported by a Weaver binary, or an empty string."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""

    output = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"\bv?(\d+\.\d+\.\d+)\b", output)
    return match.group(1) if match else ""


def _download_file(url: str, destination: Path) -> None:
    """Download a file to disk."""
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_weaver() -> Path:
    """Ensure the pinned Weaver version is available and return its binary path."""
    expected_version = _normalize_version(WEAVER_VERSION)

    system_weaver = shutil.which(_weaver_binary_name())
    if system_weaver and _weaver_version(system_weaver) == expected_version:
        return Path(system_weaver)

    cache_root = path_from_env(
        "WEAVER_CACHE",
        Path.home() / ".cache" / "otel-conformance" / "weaver",
    )
    install_dir = cache_root / WEAVER_VERSION.replace("/", "_")
    cached_binary = _find_weaver_binary(install_dir) if install_dir.exists() else None
    if cached_binary and _weaver_version(cached_binary) == expected_version:
        return cached_binary

    asset_name = _weaver_asset_name()
    download_url = (
        f"https://github.com/open-telemetry/weaver/releases/download/{WEAVER_VERSION}/{asset_name}"
    )
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=str(install_dir.parent),
        prefix=f"weaver-{expected_version}-",
    ) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / asset_name
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        print(f"=== Downloading Weaver {WEAVER_VERSION} ===")
        _download_file(download_url, archive_path)

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_dir)
        else:
            with tarfile.open(archive_path, "r:*") as archive:
                archive.extractall(extract_dir)

        extracted_binary = _find_weaver_binary(extract_dir)
        if extracted_binary is None:
            raise RuntimeError(f"Downloaded Weaver archive did not contain {_weaver_binary_name()}")

        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.copytree(extract_dir, install_dir)

    cached_binary = _find_weaver_binary(install_dir)
    if cached_binary is None:
        raise RuntimeError(f"Installed Weaver binary not found under {install_dir}")
    if sys.platform != "win32":
        cached_binary.chmod(cached_binary.stat().st_mode | 0o111)
    installed_version = _weaver_version(cached_binary)
    if installed_version != expected_version:
        raise RuntimeError(
            f"Installed Weaver version mismatch: expected {WEAVER_VERSION}, found {installed_version or 'unknown'}"
        )
    return cached_binary


def ensure_semconv_registry() -> str:
    """Ensure the pinned semconv registry is cached locally and return its model dir path."""
    registry = os.environ.get("REGISTRY")
    if registry:
        return registry

    semconv_cache_root = path_from_env(
        "SEMCONV_CACHE",
        Path.home() / ".cache" / "otel-conformance" / "semconv",
    )
    semconv_cache = semconv_cache_root / SEMCONV_VERSION.replace("/", "_")
    model_dir = semconv_cache / "model"

    if not model_dir.is_dir():
        print(f"=== Caching semantic conventions registry ({SEMCONV_VERSION}) ===")
        semconv_cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                SEMCONV_VERSION,
                "--depth",
                "1",
                "-q",
                "https://github.com/open-telemetry/semantic-conventions.git",
                str(semconv_cache),
            ],
            check=True,
        )

    return str(model_dir)


def _generate_schemas_rego(schemas: dict[str, object]) -> str:
    """Return a Rego source that defines schema constants for the live_check_advice package.

    Each constant is named _schema_<key> and holds the JSON Schema object inlined
    from the semconv repository.  Written to policies/_schemas.rego at test-run time
    (listed in .gitignore).
    """
    lines = [
        "# Auto-generated from the semconv repository.  Do not edit.",
        "# Re-generated each time prepare_advice_policies_dir() is called.",
        "package live_check_advice",
        "",
        "import rego.v1",
        "",
    ]
    for key, schema in schemas.items():
        lines.append(f"_schema_{key} := {json.dumps(schema)}")
        lines.append("")
    return "\n".join(lines)


def prepare_advice_policies_dir(semconv_registry: str) -> Path:
    """Generate _schemas.rego into the policies directory and return it.

    The directory is passed to ``weaver registry live-check --advice-policies``.
    Weaver only loads .rego files from this directory, so JSON schemas are inlined
    as Rego constants rather than loaded as OPA data documents.
    """
    semconv_root = Path(semconv_registry).parent  # model/ → semconv repo root
    docs_genai = semconv_root / "docs" / "gen-ai"

    schemas: dict[str, object] = {}
    for key, filename in _GENAI_SCHEMA_FILES.items():
        schema_path = docs_genai / filename
        if schema_path.exists():
            schemas[key] = json.loads(schema_path.read_text(encoding="utf-8"))
        else:
            print(f"Warning: GenAI schema not found: {schema_path}", file=sys.stderr)

    (POLICIES_SOURCE_DIR / "_schemas.rego").write_text(
        _generate_schemas_rego(schemas), encoding="utf-8"
    )

    return POLICIES_SOURCE_DIR
