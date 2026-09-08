"""Sweep driver: run every (prompt class x concurrency) cell against one server.

One invocation measures one server configuration, because the server is started
by hand on the GPU box. Sweeping configs means running this once per config and
letting MLflow hold the comparison.

Usage:
    # emit the serve scripts, then start one of them on the GPU box
    python -m bench.run --emit-scripts

    # measure the running server
    python -m bench.run --config-id baseline_bf16 --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from bench import metrics as metrics_mod
from bench import tracking
from bench.classes import PromptClass, select_classes
from bench.client import run_load, warmup
from bench.scenarios import read_scenarios
from bench.server import ServerConfig, load_configs, server_info, wait_for_ready, write_serve_scripts

ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "results"


def auto_requests(concurrency: int) -> int:
    """Requests per cell when not given explicitly.

    Enough waves through the pool that a p95 is not dominated by ramp-up, and
    at least 32 so the percentile has something to interpolate between.
    """
    return max(32, 4 * concurrency)


def format_cell(cls: PromptClass, concurrency: int, values: dict[str, float]) -> str:
    def g(key: str, scale: float = 1.0, digits: int = 1) -> str:
        v = values.get(key)
        return f"{v * scale:.{digits}f}" if v is not None else "-"

    line = (
        f"  {cls.id:16s} c={concurrency:<4d} "
        f"ttft_p50={g('ttft_s_p50', 1000):>8s}ms "
        f"ttft_p95={g('ttft_s_p95', 1000):>8s}ms "
        f"tpot_p50={g('tpot_s_p50', 1000):>7s}ms "
        f"out_tps={g('output_tps', 1, 1):>8s} "
        f"ok={int(values.get('requests_ok', 0))}/{int(values.get('requests_total', 0))}"
    )
    # Only present on a speculative server. Acceptance is the explanation for
    # whatever speedup the row shows, so it belongs next to it rather than only
    # in MLflow.
    if "spec_mean_accepted_length" in values:
        line += (
            f" accept={g('spec_acceptance_rate', 1, 3)}"
            f" len={g('spec_mean_accepted_length', 1, 2)}"
        )
    return line


async def run_cell(
    cls: PromptClass,
    concurrency: int,
    cfg: ServerConfig,
    base_url: str,
    args: argparse.Namespace,
) -> tuple[dict, dict, list[dict]]:
    scenarios = read_scenarios(cls.prompt_file)
    n_requests = args.requests_per_cell or auto_requests(concurrency)

    if n_requests > len(scenarios):
        print(
            f"  note: {cls.id} has {len(scenarios)} prompts for {n_requests} requests; "
            f"the set will wrap. Harmless with prefix caching off, misleading with it on.",
            file=sys.stderr,
        )

    # Counters are cumulative, so the window is differenced rather than read.
    # Gauges are sampled during the load instead, since by the time it drains
    # cache usage and queue depth have both fallen back to zero.
    before = metrics_mod.scrape(base_url)

    async with metrics_mod.GaugeSampler(base_url) as sampler:
        result = await run_load(
            scenarios,
            base_url=base_url,
            model=cfg.served_model_name,
            concurrency=concurrency,
            num_requests=n_requests,
            api_key=args.api_key,
            request_timeout=args.request_timeout,
            collect_output=args.collect_output,
        )

    after = metrics_mod.scrape(base_url)

    run_metrics = metrics_mod.aggregate(result)
    values = run_metrics.merged()
    values.update(metrics_mod.spec_decode_metrics(before, after))
    values.update(metrics_mod.preemption_delta(before, after))
    values.update(sampler.metrics())

    notes = dict(run_metrics.notes)
    notes.update(
        metrics_mod.check_prompt_lengths(result.successful, cls.input_tokens)
    )
    if not notes.get("length_capped", True):
        print(
            f"  warning: {cls.id} did not stop on length in every request "
            f"({notes.get('finish_reasons')}); ignore_eos may not be in force.",
            file=sys.stderr,
        )

    # A preempted sequence is evicted and recomputed, which inflates tail
    # latency for a reason that has nothing to do with the config under test.
    # It shows up first on a speculative server, where the draft model takes
    # memory out of the same budget the KV cache draws on.
    if values.get("preemptions", 0) > 0:
        print(
            f"  warning: {int(values['preemptions'])} preemption(s) during "
            f"{cls.id} at c={concurrency}, peak KV cache "
            f"{values.get('kv_cache_usage_peak', float('nan')):.2f}. Tail latency "
            f"here reflects cache pressure, not the config. Lower "
            f"--requests-per-cell or max_num_seqs and rerun this cell.",
            file=sys.stderr,
        )

    records = [r.to_dict() for r in result.records]
    return values, notes, records


async def main_async(args: argparse.Namespace) -> int:
    configs = load_configs()

    if args.emit_scripts:
        written = write_serve_scripts(configs)
        for path in written:
            print(f"wrote {path}")
        return 0

    if not args.config_id:
        print("--config-id is required (or use --emit-scripts)", file=sys.stderr)
        return 2
    if args.config_id not in configs:
        print(
            f"unknown config {args.config_id!r}; known: {sorted(configs)}", file=sys.stderr
        )
        return 2

    cfg = configs[args.config_id]
    base_url = args.server_url or cfg.base_url
    classes = select_classes([c.strip() for c in args.classes.split(",") if c.strip()])
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]

    if not cfg.enabled:
        print(
            f"note: config {cfg.id!r} is marked disabled in configs.yaml, which "
            f"means it is not part of the current run plan. Measuring it anyway.",
            file=sys.stderr,
        )

    print(f"config      : {cfg.id} ({cfg.name})")
    print(f"server      : {base_url}")
    print(f"classes     : {', '.join(c.id for c in classes)}")
    print(f"concurrency : {concurrencies}")
    print(f"cells       : {len(classes) * len(concurrencies)}")

    if args.dry_run:
        for cls in classes:
            for concurrency in concurrencies:
                n = args.requests_per_cell or auto_requests(concurrency)
                print(f"  would run {cls.id} at c={concurrency} for {n} requests")
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

    if args.warmup:
        print(f"warmup      : {args.warmup} requests on {classes[0].id}")
        await warmup(
            read_scenarios(classes[0].prompt_file),
            base_url=base_url,
            model=cfg.served_model_name,
            num_requests=args.warmup,
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_dir = ARTIFACT_ROOT / f"{cfg.id}-{stamp}"

    if args.mlflow:
        tracking.setup(args.experiment)

    parent_params = {
        **cfg.as_params(),
        "vllm_version": info.get("vllm_version"),
        "server_url": base_url,
        "prompt_manifest": _manifest_digest(),
    }

    summary: list[str] = []

    def emit(line: str) -> None:
        print(line)
        summary.append(line)

    if args.mlflow:
        ctx = tracking.config_run(run_name=cfg.id, params=parent_params)
    else:
        ctx = _null_context()

    with ctx:
        for cls in classes:
            for concurrency in concurrencies:
                cell_id = f"{cls.id}__c{concurrency}"
                print(f"\nrunning {cell_id} ...")
                values, notes, records = await run_cell(cls, concurrency, cfg, base_url, args)
                emit(format_cell(cls, concurrency, values))

                if args.mlflow:
                    cell_params = {
                        **cfg.as_params(),
                        **cls.as_params(),
                        "concurrency": concurrency,
                        "num_requests": len(records),
                        "ignore_eos": True,
                    }
                    with tracking.cell_run(run_name=cell_id, params=cell_params):
                        tracking.log_cell_results(
                            values, notes, records, artifact_dir, cell_id
                        )

    print("\nsummary")
    for line in summary:
        print(line)
    if args.mlflow:
        print(f"\nartifacts: {artifact_dir}")
    return 0


def _manifest_digest() -> str:
    """Tie a run to the exact prompt sets it used."""
    from bench.classes import PROMPTS_DIR

    path = PROMPTS_DIR / "manifest.json"
    if not path.exists():
        return "missing"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return ",".join(
        f"{k}:{v.get('sha256_16', '?')}" for k, v in sorted(manifest.get("classes", {}).items())
    )


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-id", help="Which entry of config/configs.yaml is running")
    parser.add_argument("--server-url", default=None, help="Overrides the config's host/port")
    parser.add_argument("--classes", default="", help="Comma separated ids; default all enabled")
    parser.add_argument("--concurrency", default="1,32")
    parser.add_argument(
        "--requests-per-cell",
        type=int,
        default=0,
        help="0 selects max(32, 4 * concurrency)",
    )
    parser.add_argument("--warmup", type=int, default=8, help="Discarded requests before measuring")
    parser.add_argument("--experiment", default=tracking.DEFAULT_EXPERIMENT)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument(
        "--collect-output",
        action="store_true",
        help="Keep generated text in memory; needed later for the quality pass",
    )
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--emit-scripts", action="store_true", help="Write scripts/serve_*.sh and exit"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
