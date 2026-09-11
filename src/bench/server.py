"""Server configs, serve-script generation, and the readiness probe.

This module never launches vLLM. The GPU box is operated by hand, so the
harness attaches to an already-running server over HTTP and instead emits the
exact command that should have started it. Keeping the launch flags in version
control rather than in shell history is what makes a run reproducible: the
script that produced a measurement sits next to the measurement.
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from bench.classes import CONFIG_DIR, REPO_ROOT

SCRIPTS_DIR = REPO_ROOT / "scripts"


@dataclass(frozen=True)
class ServerConfig:
    id: str
    name: str
    model_path: str
    served_model_name: str
    quant_method: str = "none"
    bit_width: str = "16"
    # Sparsity is a checkpoint property like quantization: vLLM reads it out of
    # the compressed-tensors block in config.json, so these are recorded for the
    # analysis rather than turned into serve flags.
    sparsity_method: str = "none"
    sparsity_ratio: float = 0.0
    sparsity_pattern: str = "none"
    spec_method: str = "none"
    num_speculative_tokens: int = 0
    draft_model_path: str = ""
    # Whether `bench.quality` scores this config. Carried in the config rather
    # than in a shell loop so the scored set is a property of the study, and so
    # a test can assert it still covers every quantization and sparsity cell.
    quality: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 256
    dtype: str = "auto"
    seed: int = 0
    extra_args: list[str] = field(default_factory=list)
    # Environment for the vLLM process, rendered as `export` lines in the serve
    # script. Some behaviour is only switchable this way (the sampler backend),
    # and it has to live next to the flags for the same reproducibility reason.
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def as_params(self) -> dict[str, object]:
        return {
            "config_id": self.id,
            "model_path": self.model_path,
            "quant_method": self.quant_method,
            "bit_width": self.bit_width,
            "sparsity_method": self.sparsity_method,
            "sparsity_ratio": self.sparsity_ratio,
            "sparsity_pattern": self.sparsity_pattern,
            "spec_method": self.spec_method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "quality_subset": self.quality,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "server_seed": self.seed,
            "server_env": " ".join(f"{k}={v}" for k, v in sorted(self.env.items())) or "none",
        }


def load_configs(path: Path | None = None) -> dict[str, ServerConfig]:
    path = path or CONFIG_DIR / "configs.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    valid = set(ServerConfig.__dataclass_fields__)

    configs: dict[str, ServerConfig] = {}
    for entry in raw["configs"]:
        merged = {**defaults, **entry}
        unknown = set(merged) - valid
        if unknown:
            raise ValueError(f"config {entry.get('id')!r} has unknown keys: {sorted(unknown)}")
        cfg = ServerConfig(**merged)
        configs[cfg.id] = cfg
    return configs


def serve_command(cfg: ServerConfig) -> list[str]:
    """The vllm serve argv for this config."""
    argv = [
        "vllm",
        "serve",
        cfg.model_path,
        "--served-model-name",
        cfg.served_model_name,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--dtype",
        cfg.dtype,
        "--seed",
        str(cfg.seed),
    ]
    if cfg.spec_method != "none":
        spec = {
            "method": cfg.spec_method,
            "model": cfg.draft_model_path,
            "num_speculative_tokens": cfg.num_speculative_tokens,
        }
        argv += ["--speculative-config", json.dumps(spec)]
    argv += list(cfg.extra_args)
    return argv


def render_serve_script(cfg: ServerConfig) -> str:
    argv = serve_command(cfg)
    # Regroup argv into flag/value pairs so each flag gets its own line. A diff
    # between two configs is then readable, which is the whole reason these are
    # generated files rather than a one-liner.
    pairs: list[str] = []
    rest = argv[3:]
    i = 0
    while i < len(rest):
        if rest[i].startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            # The speculative config is JSON, so values are shell-quoted.
            pairs.append(f"{rest[i]} {shlex.quote(rest[i + 1])}")
            i += 2
        else:
            pairs.append(rest[i])
            i += 1
    body = " \\\n    ".join([f"vllm serve {argv[2]}"] + pairs)
    exports = "".join(
        f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted(cfg.env.items())
    )

    return f"""#!/usr/bin/env bash
# Generated by bench.server. Do not edit by hand; edit config/configs.yaml.
#
# Config : {cfg.id}  ({cfg.name})
# Quant  : {cfg.quant_method} / {cfg.bit_width}-bit
# Sparse : {cfg.sparsity_method} / {cfg.sparsity_pattern} @ {cfg.sparsity_ratio}
# Spec   : {cfg.spec_method}
#
# max-model-len and gpu-memory-utilization are pinned to the same values in
# every config so that a smaller checkpoint does not silently receive a larger
# KV cache and get credited for it.
set -euo pipefail

{exports}{body}
"""


def write_serve_scripts(
    configs: dict[str, ServerConfig], out_dir: Path | None = None
) -> list[Path]:
    out_dir = out_dir or SCRIPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cfg in configs.values():
        path = out_dir / f"serve_{cfg.id}.sh"
        path.write_text(render_serve_script(cfg), encoding="utf-8", newline="\n")
        written.append(path)
    return written


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def wait_for_ready(base_url: str, timeout: float = 600.0, interval: float = 3.0) -> None:
    """Block until /health answers, or raise."""
    deadline = time.time() + timeout
    last: str = "no attempt made"
    while time.time() < deadline:
        try:
            response = httpx.get(base_url.rstrip("/") + "/health", timeout=5.0)
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(interval)
    raise TimeoutError(f"server at {base_url} not ready after {timeout:.0f}s ({last})")


def server_info(base_url: str) -> dict[str, Any]:
    """Model id and version as the server reports them.

    Recorded on every run so that a result can be traced to the checkpoint that
    produced it rather than to the config file that was supposed to.
    """
    info: dict[str, Any] = {}
    try:
        models = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=10.0).json()
        info["served_models"] = [m.get("id") for m in models.get("data", [])]
    except Exception as exc:  # noqa: BLE001
        info["served_models_error"] = str(exc)
    try:
        info["vllm_version"] = httpx.get(
            base_url.rstrip("/") + "/version", timeout=10.0
        ).json().get("version")
    except Exception:  # noqa: BLE001
        info["vllm_version"] = None
    return info


def check_context_budget(cfg: ServerConfig, concurrency: int, longest_request: int) -> str | None:
    """Reasons this config cannot honestly run a cell, or None.

    Asking for more concurrency than `max_num_seqs` does not fail. vLLM queues
    the excess, so the cell still returns numbers -- they are just the numbers
    for `max_num_seqs` concurrency with a queue in front, not for the
    concurrency the run claims to have measured.

    A request longer than `max_model_len` is rejected by vLLM rather than
    truncated, so that case surfaces as failed requests anyway; refusing up
    front costs nothing and says why before a pass is spent finding out.
    """
    if concurrency > cfg.max_num_seqs:
        return (
            f"concurrency {concurrency} exceeds --max-num-seqs {cfg.max_num_seqs}: "
            f"the excess would queue, and the cell would report queued latency "
            f"as though it were concurrent latency"
        )
    if longest_request > cfg.max_model_len:
        return (
            f"requests need up to {longest_request} tokens but --max-model-len is "
            f"{cfg.max_model_len}; raise max_model_len in config/configs.yaml"
        )
    return None
