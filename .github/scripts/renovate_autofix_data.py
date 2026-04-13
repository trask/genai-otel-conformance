#!/usr/bin/env python3
"""Regenerate committed data files for Renovate PRs from CI result artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genai_otel_conformance.data_files import generate_single_test_data


VALID_LANGUAGES = {"python", "js", "java", "dotnet"}


def _git_changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _existing_tests_for_library(tests_dir: Path, language: str, library: str) -> set[str]:
    test_names: set[str] = set()
    library_dir = tests_dir / language / library
    if language == "js":
        for test_file in sorted(library_dir.glob("test_*.ts")):
            ecosystem = test_file.stem.removeprefix("test_")
            test_names.add(f"js-{library}-{ecosystem}")
        return test_names

    for data_file in sorted(library_dir.glob("data-*.json")):
        ecosystem = data_file.stem.removeprefix("data-")
        test_names.add(f"{language}-{library}-{ecosystem}")
    return test_names


def _all_tests_for_language(tests_dir: Path, language: str) -> set[str]:
    test_names: set[str] = set()
    language_dir = tests_dir / language
    for library_dir in sorted(language_dir.iterdir()):
        if not library_dir.is_dir():
            continue
        test_names.update(_existing_tests_for_library(tests_dir, language, library_dir.name))
    return test_names


def _impacted_tests(changed_files: list[str], tests_dir: Path) -> set[str]:
    impacted: set[str] = set()

    for changed_file in changed_files:
        path = PurePosixPath(changed_file)
        parts = path.parts
        if len(parts) < 3 or parts[0] != "tests":
            continue

        language = parts[1]
        if language not in VALID_LANGUAGES:
            continue

        if language == "python":
            if len(parts) == 4 and parts[3].startswith("requirements-") and path.suffix == ".txt":
                library = parts[2]
                ecosystem = path.stem.removeprefix("requirements-")
                impacted.add(f"python-{library}-{ecosystem}")
            continue

        if language == "js":
            if len(parts) == 3 and parts[2] == "package.json":
                impacted.update(_all_tests_for_language(tests_dir, "js"))
            elif len(parts) == 4 and parts[3] == "package.json":
                impacted.update(_existing_tests_for_library(tests_dir, "js", parts[2]))
            continue

        if language == "java" and len(parts) == 4 and parts[3] == "build.gradle.kts":
            impacted.update(_existing_tests_for_library(tests_dir, "java", parts[2]))
            continue

        if language == "dotnet" and len(parts) >= 4 and path.suffix == ".csproj":
            impacted.update(_existing_tests_for_library(tests_dir, "dotnet", parts[2]))

    return impacted


def _iter_downloaded_result_dirs(artifacts_dir: Path) -> list[Path]:
    result_dirs: list[Path] = []
    for candidate in artifacts_dir.rglob("*"):
        if not candidate.is_dir() or candidate.name == "results":
            continue
        parts = candidate.parts
        if len(parts) < 4 or parts[-2] != "results":
            continue
        if parts[-4] not in VALID_LANGUAGES:
            continue
        result_dirs.append(candidate)
    return sorted(result_dirs)


def _copy_result_artifacts(artifacts_dir: Path, tests_dir: Path) -> None:
    copied = 0
    for source_dir in _iter_downloaded_result_dirs(artifacts_dir):
        language = source_dir.parts[-4]
        library = source_dir.parts[-3]
        ecosystem = source_dir.parts[-1]
        destination = tests_dir / language / library / "results" / ecosystem
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, destination)
        copied += 1
    print(f"Copied {copied} result directories from workflow artifacts")


def _update_data_file(test_name: str) -> bool:
    generated = generate_single_test_data(test_name)
    if generated is None:
        print(f"Skipping {test_name}: no parseable results found")
        return False
    if not generated.has_relevant_data:
        print(f"Skipping {test_name}: no relevant generated data")
        return False

    serialized = json.dumps(generated.data, indent=2) + "\n"
    existing = generated.path.read_text(encoding="utf-8") if generated.path.is_file() else None
    if existing == serialized:
        print(f"{test_name}: no data file changes")
        return False

    generated.path.parent.mkdir(parents=True, exist_ok=True)
    generated.path.write_text(serialized, encoding="utf-8")
    print(f"{test_name}: updated {generated.path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--base-ref", required=True)
    args = parser.parse_args()

    tests_dir = REPO_ROOT / "tests"
    changed_files = _git_changed_files(args.base_ref)
    impacted_tests = sorted(_impacted_tests(changed_files, tests_dir))
    if not impacted_tests:
        print("No impacted tests detected from changed dependency files")
        return 0

    print("Impacted tests:")
    for test_name in impacted_tests:
        print(f"- {test_name}")

    _copy_result_artifacts(args.artifacts_dir, tests_dir)

    updated_count = 0
    for test_name in impacted_tests:
        if _update_data_file(test_name):
            updated_count += 1

    print(f"Updated {updated_count} data file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())