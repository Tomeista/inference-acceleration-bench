"""Quality pass: what a config gets *wrong* in exchange for its speed.

The other half of `bench.run`. That module measures how fast a config answers
and goes to some trouble to make sure the content of the answer cannot affect
the timing -- `ignore_eos` on, output length pinned, every config decoding the
same number of steps. The consequence is that its outputs are not scoreable.
This module scores instead, and inverts nearly every one of those choices:

    bench.run                           bench.quality
    ignore_eos on, length pinned        natural stopping, truncation counted
    prompt shapes, no right answer      public benchmark items with a key
    all 7 configs                       the configs flagged `quality: true`
    output discarded                    output scored and kept per item

What it does NOT invert is the server. The same `scripts/serve_<config>.sh`
starts the same vLLM with the same `max_model_len`, `max_num_seqs` and memory
pin; prefix caching cannot change *what* a config answers, only how fast, so
there is no second script set and no second serving regime to keep in sync.

Run the reference config first. Everything else is compared against it per
item, and `agreement_with_reference` -- not accuracy -- is the metric with the
resolution to see where the damage starts.

Usage:
    python -m bench.quality --config-id baseline_bf16    # the reference, first
    python -m bench.quality --config-id w4a16_gptq
    python -m bench.quality --config-id baseline_bf16 --concurrency 1 --suffix c1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from bench import tracking
from bench.classes import REPO_ROOT
from bench.client import run_load
from bench.scoring import ItemResult, aggregate, agreement, score_records
from bench.server import (
    ServerConfig,
    check_context_budget,
    load_configs,
    probe_capabilities,
    profile_problem,
    server_info,
    wait_for_ready,
)
from bench.suites import MANIFEST_PATH, Suite, load_suite, select_suites

ARTIFACT_ROOT = REPO_ROOT / "results"

# Per-item answers, kept outside the MLflow store because they are read back by
# the *next* config's run rather than by a human. This is the join that makes
# agreement-with-reference possible, so it needs a stable path and a lifetime
# longer than one run's artifacts.
QUALITY_ROOT = ARTIFACT_ROOT / "quality"

# The config every other config is scored against. Not "the best" config -- the
# full-precision dense one, which is the only sense in which any of these has a
# correct answer to diverge from.
DEFAULT_REFERENCE = "baseline_bf16"

# Warning thresholds. Neither is fatal: both describe how to read the cell
# rather than whether it happened.
TRUNCATION_WARN = 0.05
UNPARSEABLE_WARN = 0.20

# Headroom the context guard allows for a prompt. The eval items are a few
# hundred tokens and are not length-pinned the way a prompt class is, so the
# bound is deliberately loose; what the guard really catches is a concurrency
# above `max_num_seqs`, which would queue.
PROMPT_BUDGET = 4096


def items_path(suite_id: str, config_id: str, suffix: str = "") -> Path:
    name = f"{config_id}__{suffix}" if suffix else config_id
    return QUALITY_ROOT / suite_id / f"{name}.jsonl"


def meta_path(items: Path) -> Path:
    """Sidecar describing what produced an items file.

    Exists because the reference join is the one place this pass can be
    confidently, silently wrong. `results/` is machine state -- gitignored,
    regenerated, easy to leave lying around -- so a `baseline_bf16.jsonl` from
    a smoke test against the mock, or from before an eval set was rebuilt,
    joins by scenario_id perfectly well and produces an agreement column about
    nothing. Item ids alone cannot detect that; what produced them can.
    """
    return items.with_suffix(".meta.json")


def write_items(path: Path, results: list[ItemResult], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for result in results:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    with meta_path(path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(meta, indent=2) + "\n")


def read_meta(path: Path) -> dict:
    sidecar = meta_path(path)
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def read_reference(path: Path) -> dict[str, str | None]:
    """The reference config's extracted answer per item, or {} if it has not run.

    {} rather than an error: a sweep that starts somewhere other than the
    reference still produces valid accuracy numbers, it just cannot produce the
    paired metric, and `scoring.agreement` then omits it rather than logging a
    zero that would read as total disagreement.
    """
    if not path.exists():
        return {}
    reference: dict[str, str | None] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            reference[row["scenario_id"]] = row["extracted"]
    return reference


def reference_mismatch(suite: Suite, cfg: ServerConfig, args: argparse.Namespace) -> str | None:
    """Why the reference answers for `suite` cannot be joined against `cfg`, or None.

    The join is keyed on scenario_id alone, and the suites are global, so a
    reference produced by a *different model* matches at full overlap: no
    missing items, no stale-manifest warning, and an agreement_with_reference
    that silently means "how often these two unrelated models agree" instead
    of "what compression cost this one".

    served_model_name is the right discriminator, not model_path: compression
    variants of one model deliberately have different paths but share a served
    name, while two model families share neither. Needs only the files on
    disk, so it runs before a single request is spent.
    """
    out_path = items_path(suite.id, cfg.id, args.suffix)
    reference_path = items_path(suite.id, args.reference)
    if reference_path == out_path or not read_reference(reference_path):
        return None
    recorded_model = read_meta(reference_path).get("served_model_name")
    if recorded_model is None or recorded_model == cfg.served_model_name:
        return None
    return (
        f"the reference answers in {reference_path.name} ({suite.id}) were "
        f"produced by served model {recorded_model!r}, but {cfg.id!r} serves "
        f"{cfg.served_model_name!r}. These are different models, so agreement "
        f"between them is not a quantization-damage metric. Pass "
        f"--reference <a config serving {cfg.served_model_name}>."
    )


def _manifest_digest() -> str:
    """Tie a run to the exact eval sets it scored, as run.py does for prompts."""
    if not MANIFEST_PATH.exists():
        return "missing"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return ",".join(
        f"{k}:{v.get('sha256_16', '?')}"
        for k, v in sorted((manifest.get("suites") or {}).items())
    )


def format_suite(suite: Suite, values: dict[str, float]) -> str:
    def g(key: str, digits: int = 3) -> str:
        v = values.get(key)
        return f"{v:.{digits}f}" if v is not None else "-"

    line = (
        f"  {suite.id:10s} n={int(values.get('n_items', 0)):<4d} "
        f"acc={g('accuracy')} "
        f"[{g('accuracy_ci_lo')}, {g('accuracy_ci_hi')}] "
        f"unparse={g('unparseable_rate')} "
        f"trunc={g('truncated_rate')} "
        f"rep={g('repetition_ratio')}"
    )
    if "agreement_with_reference" in values:
        line += f" agree={g('agreement_with_reference')}"
    return line


async def run_suite(
    suite: Suite,
    cfg: ServerConfig,
    base_url: str,
    args: argparse.Namespace,
) -> tuple[dict, dict, list[ItemResult]]:
    """Score one suite against one config."""
    # Before any request: a wrong reference used to be discovered only after
    # the whole suite had been generated, and the pass was then thrown away.
    if problem := reference_mismatch(suite, cfg, args):
        print(f"  refusing to score: {problem}", file=sys.stderr)
        raise SystemExit(2)

    scenarios, answers = load_suite(suite, verify=not args.no_verify)
    if args.n_items:
        # Smoke-test escape hatch. Never for a real measurement: two configs
        # scored over different item counts are not comparable, which is why
        # the run parameters record what was actually used.
        scenarios = scenarios[: args.n_items]

    started = time.perf_counter()
    result = await run_load(
        scenarios,
        base_url=base_url,
        model=cfg.served_model_name,
        concurrency=args.concurrency,
        # Every item exactly once. `run_load` cycles the set to fill the count,
        # so asking for precisely len(scenarios) is what stops an item being
        # scored twice and another not at all.
        num_requests=len(scenarios),
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        collect_output=True,
    )
    elapsed = time.perf_counter() - started

    results = score_records(result.records, answers, suite.scorer)
    values = aggregate(results)
    values.update(
        {
            "requests_ok": float(len(result.successful)),
            "requests_total": float(len(result.records)),
            "duration_s": elapsed,
        }
    )

    # Read the reference before writing our own file, so that a re-run of the
    # reference config under a --suffix compares against its earlier pass
    # instead of against the file it is in the middle of replacing. That
    # comparison is the determinism check: same weights, same prompts,
    # different batch composition.
    out_path = items_path(suite.id, cfg.id, args.suffix)
    reference_path = items_path(suite.id, args.reference)
    eval_manifest = _manifest_digest()
    stale_reference = False

    if reference_path != out_path:
        reference = read_reference(reference_path)
        reference_meta = read_meta(reference_path)
        recorded = reference_meta.get("eval_manifest")
        # A reference built against different eval bytes is comparing answers
        # to different questions. Warn rather than refuse: the accuracy in this
        # cell is still sound, and only the agreement column is affected.
        stale_reference = bool(reference) and recorded is not None and recorded != eval_manifest
        if stale_reference:
            print(
                f"  warning: the reference answers in {reference_path.name} were "
                f"produced against eval sets {recorded}, not the {eval_manifest} "
                f"being scored now. agreement_with_reference compares answers to "
                f"different questions. Re-run --config-id {args.reference} first.",
                file=sys.stderr,
            )
        values.update(agreement(results, reference))

        overlap = values.get("agreement_n")
        if overlap is not None and overlap < len(results):
            print(
                f"  warning: the reference covers only {int(overlap)} of "
                f"{len(results)} items scored here. Agreement is over the "
                f"overlap; the two runs did not see the same set.",
                file=sys.stderr,
            )

    write_items(
        out_path,
        results,
        {
            "config_id": cfg.id,
            # Which model produced these answers, so the next config to join
            # against them can tell whether that join means anything. config_id
            # alone cannot: it is the thing the caller already chose.
            "served_model_name": cfg.served_model_name,
            "model_path": cfg.model_path,
            "suite_id": suite.id,
            "concurrency": args.concurrency,
            "suffix": args.suffix,
            "n_items": len(results),
            "eval_manifest": eval_manifest,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    notes = {
        "finish_reasons": sorted({r.finish_reason or "none" for r in results}),
        "reference": args.reference,
        # Whether this cell was *supposed* to produce a paired agreement -- it is
        # every config but the reference itself. Distinct from
        # `reference_available`, which says whether it actually did: the gap
        # between the two is a reference run that is missing or has drifted, and
        # without recording the intent an absent agreement column is
        # indistinguishable from a cell that never needed one.
        "agreement_expected": reference_path != out_path,
        "reference_available": "agreement_with_reference" in values,
        "reference_stale": stale_reference,
        # Repo-relative when it sits under the repo, absolute otherwise: the
        # items store is redirectable, and a note that records where the
        # answers went must not be the thing that ends the run.
        "items_file": str(
            out_path.relative_to(REPO_ROOT)
            if out_path.is_relative_to(REPO_ROOT)
            else out_path
        ),
    }

    if values.get("requests_ok", 0) < values.get("requests_total", 0):
        failed = int(values["requests_total"] - values["requests_ok"])
        first = next((r.error for r in result.records if r.error), "no detail")
        print(
            f"  warning: {failed} request(s) failed and are not scored ({first}). "
            f"Accuracy here is over fewer items than the suite defines.",
            file=sys.stderr,
        )

    if values.get("truncated_rate", 0.0) > TRUNCATION_WARN:
        print(
            f"  warning: {values['truncated_rate']:.0%} of replies hit max_tokens "
            f"({suite.max_tokens}). A truncated reply is an unmeasured item, not a "
            f"wrong one, so this cell understates accuracy by up to that much. "
            f"Raise max_tokens for {suite.id} and re-run every config, or read "
            f"the accuracy as a lower bound.",
            file=sys.stderr,
        )

    if values.get("unparseable_rate", 0.0) > UNPARSEABLE_WARN:
        print(
            f"  warning: no answer could be extracted from "
            f"{values['unparseable_rate']:.0%} of replies. On the BF16 reference "
            f"or a lightly compressed config, suspect the scorer and read "
            f"{out_path.name}; on the most aggressive compression, this may be "
            f"the result -- instruction-following fails before accuracy does.",
            file=sys.stderr,
        )

    return values, notes, results


async def preflight(cfg: ServerConfig, suite: Suite, base_url: str, args) -> int:
    """Prove the server can be scored at all, before spending the pass.

    One assumption, and it is the mirror image of the one `bench.run` relies
    on. That module needs `ignore_eos` to be *honoured*; this one needs it to
    be absent, so that generation stops where the model would stop. Anything
    that forces length -- an `ignore_eos` injected somewhere between this
    client and the sampler, or a chat template that never lets the model end
    its turn -- produces max_tokens of text for every item, and every reply is
    then either truncated or padded past its answer. Nothing errors; the
    accuracy is just wrong, uniformly, in a way that looks like a bad
    checkpoint.
    """
    scenarios, answers = load_suite(suite, verify=not args.no_verify)
    result = await run_load(
        scenarios[:4],
        base_url=base_url,
        model=cfg.served_model_name,
        concurrency=1,
        num_requests=4,
        collect_output=True,
    )

    ok = result.successful
    if not ok:
        first = next((r.error for r in result.records if r.error), "no detail")
        print(f"preflight FAILED: no request succeeded ({first})", file=sys.stderr)
        return 1

    results = score_records(result.records, answers, suite.scorer)
    reasons = sorted({r.finish_reason or "none" for r in ok})
    extracted = [r.extracted for r in results]
    print(f"  finish reasons      : {reasons}")
    print(f"  extracted answers   : {extracted}")
    print(f"  expected answers    : {[r.expected for r in results]}")

    problems: list[str] = []
    if reasons == ["length"]:
        problems.append(
            "every reply stopped at max_tokens. Something is forcing length "
            "(ignore_eos), or the chat template is not letting the model end "
            "its turn, so nothing here stops where the model would stop and no "
            "reply can be scored honestly. Start the server from "
            f"scripts/serve_{cfg.id}.sh and check what sits in front of it."
        )
    if all(e is None for e in extracted):
        problems.append(
            "no answer could be extracted from any of the four replies. Either "
            "the chat template is not rendering the prompt as built, or this "
            "config cannot follow the format at all. Read the replies before "
            "trusting a full pass."
        )

    if problems:
        print("\npreflight FAILED:", file=sys.stderr)
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}", file=sys.stderr)
        return 1

    print("\npreflight OK")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    configs = load_configs()
    if not args.config_id:
        print("--config-id is required", file=sys.stderr)
        return 2
    if args.config_id not in configs:
        print(f"unknown config {args.config_id!r}; known: {sorted(configs)}", file=sys.stderr)
        return 2

    cfg = configs[args.config_id]
    if not cfg.quality and not args.force:
        subset = sorted(c.id for c in configs.values() if c.quality)
        print(
            f"{cfg.id} is not in the quality subset ({', '.join(subset)}). The "
            f"subset is pinned in config/configs.yaml so that the scored configs "
            f"are a property of the study rather than of whoever typed the "
            f"loop. Use --force to score it anyway.",
            file=sys.stderr,
        )
        return 2

    base_url = args.server_url or cfg.base_url
    requested = [s.strip() for s in args.suites.split(",") if s.strip()]
    suites = select_suites(requested)

    # A config lists the serve profiles it is scored under, and a suite runs
    # only on configs that list its profile. That is what keeps bfcl_ast to the
    # four core configs: the others have no `tools` server by design.
    outside = [s.id for s in suites if s.profile not in cfg.profiles]
    if outside:
        if requested:
            print(
                f"cannot run {', '.join(outside)} on {cfg.id}: it needs a serve "
                f"profile {cfg.id} does not list ({cfg.profiles}) in config/configs.yaml",
                file=sys.stderr,
            )
            return 2
        suites = [s for s in suites if s.id not in outside]
        print(f"not scored on {cfg.id} by design: {', '.join(outside)}", file=sys.stderr)

    for suite in suites:
        if problem := check_context_budget(
            cfg, args.concurrency, PROMPT_BUDGET + suite.max_tokens
        ):
            print(f"cannot run {suite.id}: {problem}", file=sys.stderr)
            return 2
        if problem := reference_mismatch(suite, cfg, args):
            print(f"cannot run {suite.id}: {problem}", file=sys.stderr)
            return 2

    print(f"config      : {cfg.id} ({cfg.name})")
    print(
        f"weights     : {cfg.quant_method} / {cfg.bit_width}-bit, "
        f"sparsity {cfg.sparsity_pattern}, speculation {cfg.spec_method}"
    )
    print(f"server      : {base_url}")
    print(f"suites      : {', '.join(s.id for s in suites)}")
    print(f"concurrency : {args.concurrency}")
    print(f"reference   : {args.reference}" + ("  (this config)" if args.reference == cfg.id else ""))

    if args.dry_run:
        for suite in suites:
            n = args.n_items or suite.n_items
            print(f"  would score {suite.id} over {n} items at c={args.concurrency}")
        return 0

    wait_for_ready(base_url, timeout=args.ready_timeout)
    info = server_info(base_url)
    print(f"vllm        : {info.get('vllm_version')}  models={info.get('served_models')}")

    if cfg.served_model_name not in (info.get("served_models") or []):
        print(
            f"warning: served model {info.get('served_models')} does not include "
            f"{cfg.served_model_name!r} from config {cfg.id!r}. The running server "
            f"may not be the one this config describes.",
            file=sys.stderr,
        )

    # Each suite names the serve profile it needs, and a server started from
    # the wrong script still answers -- every BFCL request would come back 400
    # from a base server, leaving an empty cell. Asked of the server itself,
    # since nothing else knows which script started it.
    caps = probe_capabilities(base_url, cfg.served_model_name)
    unservable = {
        s.id: p for s in suites if (p := profile_problem(s.profile, caps))
    }
    if unservable:
        for suite_id, problem in unservable.items():
            print(f"cannot run {suite_id} on this server: {problem}", file=sys.stderr)
        if requested:
            return 2
        # A default selection spans profiles, and no one server covers them
        # all, so the suites this server cannot run are skipped by name.
        suites = [s for s in suites if s.id not in unservable]
        print(
            f"skipping {', '.join(unservable)}: not requested by name, and this "
            f"server cannot run them. Score them from the matching serve script.",
            file=sys.stderr,
        )
        if not suites:
            return 2

    if args.preflight:
        print("\npreflight ...")
        return await preflight(cfg, suites[0], base_url, args)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_dir = ARTIFACT_ROOT / f"{cfg.id}-quality-{stamp}"

    if args.mlflow:
        tracking.setup(args.experiment, tracking_uri=args.tracking_uri)

    parent_params = {
        **cfg.as_params(),
        "vllm_version": info.get("vllm_version"),
        "server_url": base_url,
        "eval_manifest": _manifest_digest(),
        "reference": args.reference,
    }

    summary: list[str] = []
    ctx = (
        tracking.config_run(
            run_name=f"{cfg.id}-quality",
            params=parent_params,
            # `pass` separates this parent from the speed sweep's parent for the
            # same config. The cells are already separated by carrying suite_id
            # instead of class_id, which is what keeps bench.report's speed
            # tables from ever seeing them.
            tags={"engine": "vllm", "pass": "quality"},
        )
        if args.mlflow
        else _null_context()
    )

    with ctx:
        for suite in suites:
            cell_id = f"{suite.id}__c{args.concurrency}"
            print(f"\nscoring {cell_id} ...")
            values, notes, results = await run_suite(suite, cfg, base_url, args)
            line = format_suite(suite, values)
            print(line)
            summary.append(line)

            if args.mlflow:
                cell_params = {
                    **cfg.as_params(),
                    **suite.as_params(),
                    "concurrency": args.concurrency,
                    "n_items_run": len(results),
                    "reference": args.reference,
                    "suffix": args.suffix,
                    "ignore_eos": False,
                }
                with tracking.cell_run(run_name=cell_id, params=cell_params):
                    tracking.log_cell_results(
                        values,
                        notes,
                        [r.to_dict() for r in results],
                        artifact_dir,
                        cell_id,
                        # Every config but the reference is meant to carry a
                        # paired agreement. Its absence means the reference run
                        # is missing or scored different bytes, which is a
                        # broken comparison rather than a cell with less to say.
                        expected=(
                            ("agreement_with_reference",)
                            if notes.get("agreement_expected")
                            else ()
                        ),
                    )

    print("\nsummary")
    for line in summary:
        print(line)
    if args.mlflow:
        print(f"\nartifacts: {artifact_dir}")
    return 0


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-id", help="Which entry of config/configs.yaml is running")
    parser.add_argument("--server-url", default=None, help="Overrides the config's host/port")
    parser.add_argument("--suites", default="", help="Comma separated ids; default all enabled")
    # One value, not a list. Scoring at two concurrencies would produce two
    # accuracies for one config that differ only by batch numerics, and nothing
    # downstream would know which to plot.
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Pinned across configs; 32 is the speed sweep's upper level, chosen for wall clock",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Config whose per-item answers every other config is compared against",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Tag this run's items file, e.g. --suffix c1 for the determinism check",
    )
    parser.add_argument(
        "--n-items", type=int, default=0, help="Truncate each suite; smoke tests only"
    )
    parser.add_argument("--experiment", default=tracking.DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Defaults to MLFLOW_TRACKING_URI, else mlflow.db at the repo root",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check that generation stops naturally and parses, then exit",
    )
    parser.add_argument(
        "--force", action="store_true", help="Score a config not in the quality subset"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the eval-set digest check (for a locally modified set)",
    )
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
