"""Loader for config/suites.yaml and the frozen eval sets beside it.

The quality-pass analogue of `classes.py`: a suite is to scoring what a prompt
class is to timing.

Why the answer key is a separate file
-------------------------------------
The obvious design is an `answer` field on `Scenario`. It is not used: that
would put the answer on the object the load client turns into requests, and
change the frozen Scenario format that `prompts/` and the speed sweep share. So
`evals/x.jsonl` is exactly that format -- `read_scenarios` loads it unchanged --
and `evals/x.key.jsonl` carries the answers alongside.

The cost of two files is that they can drift apart, and a key that has drifted
scores noise while looking perfectly healthy. `load_suite` therefore refuses to
return a suite whose two files disagree on scenario ids, and the digest check
below fails on either file changing at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from bench.classes import CONFIG_DIR, REPO_ROOT
from bench.scenarios import Scenario, read_scenarios

EVALS_DIR = REPO_ROOT / "evals"
MANIFEST_PATH = EVALS_DIR / "manifest.json"


@dataclass(frozen=True)
class Suite:
    """One benchmark, frozen: which items, how they are asked, how scored."""

    id: str
    name: str
    source: str
    dataset: str
    revision: str
    parquet: str
    max_tokens: int
    scorer: str
    description: str = ""
    license: str = ""
    temperature: float = 0.0
    n_items: int = 250
    enabled: bool = True

    @property
    def prompt_file(self) -> Path:
        return EVALS_DIR / f"{self.id}.jsonl"

    @property
    def key_file(self) -> Path:
        return EVALS_DIR / f"{self.id}.key.jsonl"

    def as_params(self) -> dict[str, object]:
        """What a quality cell records about the suite it ran.

        `suite_id` rather than `class_id` is load-bearing: `report.load_cells`
        selects speed cells on `params.class_id.notna()`, so naming this field
        `class_id` would silently pool accuracy runs into the speed tables.
        """
        return {
            "suite_id": self.id,
            "suite_source": self.source,
            "dataset": self.dataset,
            "dataset_revision": self.revision,
            "n_items": self.n_items,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "scorer": self.scorer,
        }


def load_suites(path: Path | None = None) -> dict[str, Suite]:
    """Load every suite, enabled or not, keyed by id."""
    path = path or CONFIG_DIR / "suites.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    valid = set(Suite.__dataclass_fields__)

    suites: dict[str, Suite] = {}
    for entry in raw["suites"]:
        merged = {**defaults, **entry}
        unknown = set(merged) - valid
        if unknown:
            raise ValueError(f"suite {entry.get('id')!r} has unknown keys: {sorted(unknown)}")
        suite = Suite(**merged)
        if suite.id in suites:
            raise ValueError(f"duplicate suite id {suite.id!r}")
        suites[suite.id] = suite
    return suites


def select_suites(requested: list[str] | None = None, path: Path | None = None) -> list[Suite]:
    """Resolve a CLI --suites selection; all enabled suites when empty."""
    suites = load_suites(path)
    if not requested:
        return [s for s in suites.values() if s.enabled]
    missing = [r for r in requested if r not in suites]
    if missing:
        raise KeyError(f"unknown suite id(s) {missing}; known: {sorted(suites)}")
    return [suites[r] for r in requested]


def digest(path: Path) -> str:
    """The same 16-hex-character convention prompts/manifest.json uses."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def check_digests(suite: Suite) -> None:
    """Refuse a suite whose frozen files are not the bytes that were built.

    Enforced here at run time as well as in the test suite, because a
    mid-sweep edit to an eval set would otherwise make two configs incomparable
    while every run still succeeded.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"no {MANIFEST_PATH}. The frozen eval sets are built by "
            f"`python -m bench.build_evals` and committed; a checkout without "
            f"them cannot run the quality pass."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded = (manifest.get("suites") or {}).get(suite.id)
    if not recorded:
        raise KeyError(f"{suite.id} is not in {MANIFEST_PATH}; rebuild the eval sets")

    for path, key in ((suite.prompt_file, "sha256_16"), (suite.key_file, "key_sha256_16")):
        if not path.exists():
            raise FileNotFoundError(f"{path} is in the manifest but not on disk")
        actual = digest(path)
        if actual != recorded[key]:
            raise ValueError(
                f"{path.name} is not the frozen set: manifest says "
                f"{recorded[key]}, file is {actual}. Configs measured against "
                f"different bytes are not comparable."
            )


def read_key(suite: Suite) -> dict[str, str]:
    """scenario_id -> gold answer, already in the scorer's normalized form."""
    answers: dict[str, str] = {}
    with suite.key_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            answers[row["scenario_id"]] = row["answer"]
    return answers


def load_suite(suite: Suite, *, verify: bool = True) -> tuple[list[Scenario], dict[str, str]]:
    """The items and their answers, checked against each other.

    The check is the point. Two files that have drifted apart still load, still
    run, and still produce an accuracy -- of noise.
    """
    if verify:
        check_digests(suite)
    scenarios = read_scenarios(suite.prompt_file)
    answers = read_key(suite)

    prompt_ids = {s.scenario_id for s in scenarios}
    key_ids = set(answers)
    if prompt_ids != key_ids:
        only_prompts = sorted(prompt_ids - key_ids)[:3]
        only_key = sorted(key_ids - prompt_ids)[:3]
        raise ValueError(
            f"{suite.id}: prompt file and key file describe different items "
            f"({len(prompt_ids)} vs {len(key_ids)}). "
            f"Only in prompts: {only_prompts}; only in key: {only_key}"
        )
    return scenarios, answers
