"""Mapping between test names, data files, and result directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestLocation:
    lang: str
    library: str
    ecosystem: str

    @property
    def test_name(self) -> str:
        return f"{self.lang}-{self.library}-{self.ecosystem}"

    def data_file(self, tests_dir: Path) -> Path:
        return tests_dir / self.lang / self.library / f"data-{self.ecosystem}.json"

    def results_dir(self, tests_dir: Path) -> Path:
        return tests_dir / self.lang / self.library / "results" / self.ecosystem

    @classmethod
    def from_test_name(cls, test_name: str) -> TestLocation:
        try:
            lang, rest = test_name.split("-", 1)
            library, ecosystem = rest.rsplit("-", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid test name: {test_name}") from exc

        if not lang or not library or not ecosystem:
            raise ValueError(f"Invalid test name: {test_name}")

        return cls(lang=lang, library=library, ecosystem=ecosystem)

    @classmethod
    def from_results_dir(cls, result_dir: Path, tests_dir: Path) -> TestLocation:
        relative = result_dir.relative_to(tests_dir)
        lang, library, _, ecosystem = relative.parts
        return cls(lang=lang, library=library, ecosystem=ecosystem)

    @classmethod
    def from_data_file(cls, data_file: Path, tests_dir: Path) -> TestLocation:
        relative = data_file.relative_to(tests_dir)
        if len(relative.parts) != 3:
            raise ValueError(
                "Expected data file under tests/<lang>/<library>/data-<ecosystem>.json, "
                f"got {relative}"
            )
        lang, library, data_name = relative.parts
        if not data_name.startswith("data-"):
            raise ValueError(f"Expected data file name starting with 'data-': {relative}")
        return cls(
            lang=lang,
            library=library,
            ecosystem=Path(data_name).stem.removeprefix("data-"),
        )