"""MLflow logging.

Structure: one parent run per server configuration, one nested child run per
(prompt class, concurrency) cell. That shape is what makes the comparison
queryable later. A single flat run per cell would force every plot to
reconstruct which cells belonged to the same server, and the whole study is
about differences between servers holding the cell fixed.

Raw per-request records are attached to each child run as an artifact. The
aggregates answer the questions we know to ask now; the records are what a
question we have not thought of yet will need.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import mlflow

DEFAULT_EXPERIMENT = "inference-acceleration"

# This repository's own store, resolved against the repo rather than the
# working directory, so the speed sweep, the quality pass and `bench.report`
# all read and write the same file wherever they are launched from. Sqlite
# rather than a directory of files: current MLflow refuses the filesystem
# backend outright ("in maintenance mode").
DEFAULT_TRACKING_URI = (
    "sqlite:///" + (Path(__file__).resolve().parents[2] / "mlflow.db").as_posix()
)


def _drop_reason(value: Any) -> str | None:
    """Why MLflow will not store this value, or None if it will.

    The single definition of the rule, so `_clean_metrics` and the absence
    record below can never disagree about what was dropped.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        # Checked before the float conversion: `bool` subclasses `int`, so True
        # would otherwise coerce to 1.0 and be stored as a number.
        return "bool"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "not_numeric"
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf"
    return None


def _clean_metrics(values: dict[str, Any]) -> dict[str, float]:
    """Drop anything MLflow cannot store as a metric.

    NaN and inf arrive whenever a cell produced no successful request. Logging
    them raises, and logging them as 0.0 would be worse than not logging them,
    because a zero p95 reads as an extraordinarily fast run.

    Silent by necessity -- but not unrecorded: `describe_absences` reports what
    this refused, and `log_cell_results` writes it beside the metrics.
    """
    return {
        key: float(value)
        for key, value in values.items()
        if _drop_reason(value) is None
    }


def describe_absences(
    values: dict[str, Any], expected: Iterable[str] = ()
) -> dict[str, str]:
    """Which metrics will not be in the store for this cell, and why.

    The reason this exists. A metric can be missing from a cell for two
    completely different reasons, and after the fact they look identical:

      by design  the cell could not compute it, and the repo's convention is
                 absent rather than zero -- a baseline server publishes no
                 speculative counters, and logging acceptance 0.0 would read as
                 total rejection rather than as "no draft model"
      by bug     it should have been there. The counter names have moved
                 between vLLM versions before, so a speculative run can lose its
                 acceptance column while every request succeeds and every other
                 number lands

    Nothing downstream can tell those apart from a hole in the store, which is
    how a missing number survives review. `expected` is what closes the gap: the
    caller says what this cell *should* have produced given its configuration,
    and anything named there but absent is recorded as `expected_but_absent`
    rather than quietly not mentioned.
    """
    out: dict[str, str] = {}
    for key, value in values.items():
        reason = _drop_reason(value)
        if reason is not None:
            out[key] = reason
    for key in expected:
        if key not in values:
            out[key] = "expected_but_absent"
    return out


def _clean_params(values: dict[str, Any]) -> dict[str, str]:
    return {k: ("" if v is None else str(v)) for k, v in values.items()}


def environment_tags() -> dict[str, str]:
    """Host and driver facts that explain a result months later."""
    tags: dict[str, str] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            lines = [ln.strip() for ln in out.stdout.strip().splitlines()]
            tags["gpu"] = lines[0]
            tags["gpu_count"] = str(len(lines))
    except Exception:  # noqa: BLE001 - absent on the workstation, present on the server
        tags["gpu"] = "unavailable"
    return tags


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """An explicit URI, else MLFLOW_TRACKING_URI, else this repo's mlflow.db."""
    return tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def setup(experiment: str = DEFAULT_EXPERIMENT, tracking_uri: str | None = None) -> None:
    """Point MLflow at its store.

    Defaults to `mlflow.db` at the repository root. Set MLFLOW_TRACKING_URI, or
    pass --tracking-uri, to push to a shared server without touching this code.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment)


@contextmanager
def config_run(
    run_name: str, params: dict[str, Any], tags: dict[str, str] | None = None
) -> Iterator[Any]:
    """Parent run for one server configuration."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**environment_tags(), **(tags or {})})
        mlflow.log_params(_clean_params(params))
        yield run


@contextmanager
def cell_run(run_name: str, params: dict[str, Any]) -> Iterator[Any]:
    """Nested run for one (class, concurrency) cell."""
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params(_clean_params(params))
        yield run


def log_cell_results(
    metrics: dict[str, Any],
    notes: dict[str, Any],
    records: list[dict[str, Any]],
    artifact_dir: Path,
    cell_id: str,
    expected: Iterable[str] = (),
) -> None:
    mlflow.log_metrics(_clean_metrics(metrics))

    # What is NOT in the store, recorded next to what is. Without this a hole is
    # indistinguishable from a metric that was never meant to exist, and the
    # only way to find out is to remember which it was.
    absences = describe_absences(metrics, expected)
    if absences:
        notes = {**notes, "metrics_absent": absences}

    # An absence the caller said to expect is the one worth interrupting for:
    # it means the cell was configured to produce a number and did not. Printed
    # as well as tagged, because nobody reads tags during a sweep.
    missing = sorted(k for k, v in absences.items() if v == "expected_but_absent")
    if missing:
        print(
            f"  warning: {cell_id} was expected to record {missing} and did not. "
            f"On a speculative config this usually means the engine's counter "
            f"names have moved; check `metrics.COUNTERS` against this vLLM "
            f"version before reading the cell as a result.",
            file=sys.stderr,
        )

    # Notes are strings and booleans (finish reasons, validity checks), so they
    # are tags rather than metrics.
    mlflow.set_tags({f"note.{k}": json.dumps(v) for k, v in notes.items()})

    artifact_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / f"{cell_id}_requests.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    mlflow.log_artifact(str(records_path), artifact_path="requests")
