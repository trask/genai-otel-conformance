"""Ecosystem, library, and dependency version metadata loaded from test directories."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from genai_otel_conformance import TESTS_DIR

LANGUAGE_DISPLAY_NAMES = {"python": "Python", "java": "Java", "js": "JS", "dotnet": "C#"}
LANGUAGE_SLUGS = {display: slug for slug, display in LANGUAGE_DISPLAY_NAMES.items()}


class EcosystemInfo(NamedTuple):
    display: dict[str, str]
    repos: dict[tuple[str, str], str]


def _load_ecosystems() -> EcosystemInfo:
    """Load ecosystem definitions from tests/ecosystems.json."""
    eco_file = TESTS_DIR / "ecosystems.json"
    data = json.loads(eco_file.read_text(encoding="utf-8"))
    display: dict[str, str] = {}
    repos: dict[tuple[str, str], str] = {}
    for eco, info in data.items():
        display[eco] = info.get("display_name", eco)
        for lang_slug, repo in info.get("repos", {}).items():
            lang_display = LANGUAGE_DISPLAY_NAMES.get(lang_slug, lang_slug)
            repos[(eco, lang_display)] = repo
    return EcosystemInfo(display, repos)

_ECOSYSTEM_INFO = _load_ecosystems()
ECOSYSTEM_DISPLAY = _ECOSYSTEM_INFO.display
ECOSYSTEM_REPOS = _ECOSYSTEM_INFO.repos


class LibraryInfo(NamedTuple):
    display_names: dict[str, str]
    native_repos: dict[tuple[str, str], str]


def _discover_library_metadata() -> LibraryInfo:
    """Scan metadata.json files for library display names and native repos."""
    names: dict[str, str] = {}
    repos: dict[tuple[str, str], str] = {}
    if not TESTS_DIR.is_dir():
        return LibraryInfo(names, repos)
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
            data = json.loads(meta.read_text(encoding="utf-8"))
            if slug not in names and "display_name" in data:
                names[slug] = data["display_name"]
            if "repo" in data:
                repos[(lang_dir.name, slug)] = data["repo"]
    return LibraryInfo(names, repos)


_LIBRARY_INFO = _discover_library_metadata()
LIBRARY_DISPLAY_NAMES = _LIBRARY_INFO.display_names
NATIVE_REPOS = _LIBRARY_INFO.native_repos


def library_display_name(slug: str) -> str:
    """Return the human-readable display name for a library slug."""
    return LIBRARY_DISPLAY_NAMES.get(slug, slug)


@lru_cache(maxsize=None)
def _load_test_metadata(lang: str, library: str) -> dict:
    """Load metadata.json for a test directory."""
    meta_file = TESTS_DIR / lang / library / "metadata.json"
    if not meta_file.is_file():
        return {}
    return json.loads(meta_file.read_text(encoding="utf-8"))


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


def _read_js_dependency_versions(test_dir: Path, _ecosystem: str) -> dict[str, str]:
    pkg_file = test_dir / "package.json"
    if not pkg_file.exists():
        return {}
    data = json.loads(pkg_file.read_text(encoding="utf-8"))
    return dict(data.get("dependencies", {}))


def _read_java_dependency_versions(test_dir: Path, _ecosystem: str) -> dict[str, str]:
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


def _read_dotnet_dependency_versions(test_dir: Path, _ecosystem: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for csproj in test_dir.glob("*.csproj"):
        content = csproj.read_text(encoding="utf-8")
        for match in re.finditer(
            r'PackageReference\s+Include="([^"]+)"\s+Version="([^"]+)"',
            content,
        ):
            versions[match.group(1)] = match.group(2)
    return versions


DependencyVersionReader = Callable[[Path, str], dict[str, str]]


DEPENDENCY_VERSION_READERS: dict[str, DependencyVersionReader] = {
    "python": _read_python_dependency_versions,
    "js": _read_js_dependency_versions,
    "java": _read_java_dependency_versions,
    "dotnet": _read_dotnet_dependency_versions,
}


def _read_deps_from_test_dir(lang: str, library: str, ecosystem: str) -> dict[str, str]:
    test_dir = TESTS_DIR / lang / library
    reader = DEPENDENCY_VERSION_READERS.get(lang)
    if reader is None:
        return {}
    return reader(test_dir, ecosystem)


def extract_version_from_deps(lang: str, library: str, ecosystem: str) -> str:
    """Extract the display version from checked-in dependency files."""
    versions = _read_deps_from_test_dir(lang, library, ecosystem)
    package_name = _version_package_from_metadata(lang, library, ecosystem)
    if not package_name:
        return ""
    return versions.get(package_name, "")
