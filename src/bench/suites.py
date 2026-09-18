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
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bench.classes import CONFIG_DIR, REPO_ROOT
from bench.scenarios import Scenario, read_scenarios

EVALS_DIR = REPO_ROOT / "evals"
MANIFEST_PATH = EVALS_DIR / "manifest.json"


# How a suite's items came to exist. The reproducibility claim is the same in
# all three cases -- a rebuild reproduces the committed bytes or fails loudly --
# but what has to be pinned to make that true differs.
#
#   huggingface  a dataset repo, a commit, a file inside it
#   repository   a git repo, a commit, a path inside it. The Gorilla data files
#                are not on the Hub
#   synthetic    no upstream to pin. Generated here, so the pin is the generator
#                name, the build seed, and a digest of the corpus it drew from
PROVENANCE = ("huggingface", "repository", "synthetic")


@dataclass(frozen=True)
class Suite:
    """One benchmark, frozen: which items, how they are asked, how scored."""

    id: str
    name: str
    source: str
    max_tokens: int
    scorer: str

    # -- provenance -------------------------------------------------------
    # `dataset`/`revision`/`parquet` read as "source, pin, path within source"
    # and carry both fetched kinds: an HF dataset id or a git URL, a 40-hex
    # commit either way, and the file inside. Reusing them rather than adding a
    # parallel set is deliberate -- the reproducibility check is identical, and
    # two near-identical field triples would drift apart.
    provenance: str = "huggingface"
    dataset: str = ""
    revision: str = ""
    parquet: str = ""
    # Synthetic suites only. `generator` names the function that emitted the
    # items; `corpus_digest` pins the material it drew from, because a suite
    # generated at a fixed seed from changed source text is a changed suite.
    generator: str = ""
    corpus_digest: str = ""
    # Further files the builder fetches beside `parquet`, relative to the same
    # pinned commit. BFCL is split across a file per category plus a separate
    # answer key for each, and a suite that fetches seven files must record
    # seven paths or its provenance is a third of the truth.
    extra_sources: list[str] = field(default_factory=list)

    # -- how it is asked --------------------------------------------------
    # The serve profile this suite requires. `bench.quality` refuses to score a
    # suite against a server that is not running it, the same class of guard as
    # the served_model_name check: a complete, plausible, mislabelled result is
    # worse than a failure.
    profile: str = "base"
    description: str = ""
    license: str = ""
    temperature: float = 0.0
    # Sampling, flat rather than a nested dict so the dataclass stays frozen and
    # hashable. The defaults are greedy, which is what every suite but the
    # thinking arm wants: two configs must differ because their weights differ,
    # not because their samplers rolled apart.
    top_k: int = 1
    top_p: float | None = None
    thinking: bool = False
    # Only the thinking arm sets this above 1. Everything else is greedy, and a
    # second greedy pass would measure batch-composition nondeterminism -- which
    # the noise-floor protocol measures deliberately instead, via --suffix.
    seeds: int = 1
    n_items: int = 250
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE:
            raise ValueError(
                f"suite {self.id!r}: provenance {self.provenance!r} is not one of "
                f"{list(PROVENANCE)}"
            )
        if self.provenance == "synthetic":
            if not self.generator:
                raise ValueError(f"suite {self.id!r} is synthetic but names no generator")
        elif len(self.revision) != 40:
            raise ValueError(
                f"suite {self.id!r} is {self.provenance} but its revision "
                f"{self.revision!r} is not a 40-character commit. A branch name "
                f"would let the benchmark change underneath the study."
            )

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
            "provenance": self.provenance,
            "dataset": self.dataset,
            "dataset_revision": self.revision,
            "extra_sources": ",".join(self.extra_sources) or "none",
            "generator": self.generator,
            "corpus_digest": self.corpus_digest,
            "n_items": self.n_items,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "thinking": self.thinking,
            "seeds": self.seeds,
            "serve_profile": self.profile,
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


def read_key(suite: Suite) -> dict[str, dict]:
    """scenario_id -> the whole key row.

    The whole row rather than just `answer`, because a scorer is handed the row:
    `answer` is the display string and the agreement join's gold, but a
    structured suite's real gold data -- instruction specs, acceptable calls,
    unit tests -- lives in `meta` and does not fit in a string.
    """
    rows: dict[str, dict] = {}
    with suite.key_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["scenario_id"]] = row
    return rows


def read_answers(suite: Suite) -> dict[str, str]:
    """scenario_id -> gold answer only, for callers that want the string."""
    return {sid: row["answer"] for sid, row in read_key(suite).items()}


def load_suite(suite: Suite, *, verify: bool = True) -> tuple[list[Scenario], dict[str, dict]]:
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
