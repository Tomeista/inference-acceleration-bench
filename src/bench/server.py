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


# --------------------------------------------------------------------------
# serve profiles
# --------------------------------------------------------------------------
#
# A profile is a serving variant of a config: identical weights, different
# serve flags. Three suites need flags the speed sweep must never see, and the
# alternative -- a second script set someone remembers to launch -- is exactly
# the kind of thing that produces a complete, plausible, mislabelled result.
#
# `speed_safe` is the load-bearing field. Under the hermes tool-call parser,
# vLLM buffers tokens until it can name the function, so TTFT measures the
# parser rather than the model (p95 16 s on the GB10) and the request ends at
# the call regardless of ignore_eos, so output length is no longer pinned.
# A latency number from that profile would be a measurement of the parser.


@dataclass(frozen=True)
class Profile:
    """One serving variant. `base` is what every existing script already is."""

    id: str
    description: str
    # Appended after the config's own extra_args, so a config can still add to
    # them without having to know which profile it will be served under.
    extra_args: tuple[str, ...] = ()
    # Overrides the config's pin. Only `long` uses it, and it is reported as its
    # own configuration rather than pooled with the 16384 numbers: a bigger
    # window means a different KV cache size, which is the thing the pin exists
    # to hold constant.
    max_model_len: int | None = None
    speed_safe: bool = True


PROFILES: dict[str, Profile] = {
    "base": Profile(
        id="base",
        description="The pinned serving regime. The only profile the speed sweep may use.",
    ),
    "tools": Profile(
        id="tools",
        description=(
            "Native function-calling, for the BFCL suites. The parser under test "
            "is then the one that would be deployed."
        ),
        extra_args=("--enable-auto-tool-choice", "--tool-call-parser", "hermes"),
        speed_safe=False,
    ),
    "long": Profile(
        id="long",
        description=(
            "32k context for RULER. Qwen3-8B's native window, so no YaRN: static "
            "rope scaling would confound the KV-cache effect with its own."
        ),
        max_model_len=32768,
        speed_safe=False,
    ),
}


def get_profile(profile_id: str) -> Profile:
    if profile_id not in PROFILES:
        raise KeyError(f"unknown serve profile {profile_id!r}; known: {sorted(PROFILES)}")
    return PROFILES[profile_id]


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
    # The KV-cache axis, first-class rather than buried in extra_args because it
    # is one of the study's two compression variables and every cell has to
    # record which setting produced it. vLLM offers fp8_e4m3/fp8_e5m2 and
    # nothing lower: there is no INT4 or INT2 KV path, so the aggressive arm the
    # literature describes cannot be run here and is reported as a tooling
    # limit rather than as a result.
    kv_cache_dtype: str = "auto"
    seed: int = 0
    # Which serving variants this config needs scripts for. `base` always; a
    # config only gets `tools` or `long` if a suite will actually be run against
    # it there, so `--emit-scripts` does not litter the directory with servers
    # nobody starts.
    profiles: list[str] = field(default_factory=lambda: ["base"])
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
            "kv_cache_dtype": self.kv_cache_dtype,
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
        # Overwriting silently is the worst of the options: a copy-pasted entry
        # that kept its old id would vanish from the run plan, from
        # --emit-scripts, and from the report's config ordering, with nothing
        # anywhere saying so. classes.py and suites.py both refuse; so does this.
        if cfg.id in configs:
            raise ValueError(f"duplicate config id {cfg.id!r} in {path}")
        configs[cfg.id] = cfg
    return configs


def effective_max_model_len(cfg: ServerConfig, profile: str = "base") -> int:
    """The context window this config is actually served with under `profile`."""
    return get_profile(profile).max_model_len or cfg.max_model_len


def serve_command(cfg: ServerConfig, profile: str = "base") -> list[str]:
    """The vllm serve argv for this config under one serving profile."""
    prof = get_profile(profile)
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
        str(effective_max_model_len(cfg, profile)),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--dtype",
        cfg.dtype,
        "--seed",
        str(cfg.seed),
    ]
    # Appended only when set, so the base profile of every existing config
    # renders exactly the script that is already committed next to its
    # measurements. A test asserts that.
    if cfg.kv_cache_dtype != "auto":
        argv += ["--kv-cache-dtype", cfg.kv_cache_dtype]
    if cfg.spec_method != "none":
        spec = {
            "method": cfg.spec_method,
            "model": cfg.draft_model_path,
            "num_speculative_tokens": cfg.num_speculative_tokens,
        }
        argv += ["--speculative-config", json.dumps(spec)]
    argv += list(cfg.extra_args)
    argv += list(prof.extra_args)
    return argv


def script_name(cfg: ServerConfig, profile: str = "base") -> str:
    """`serve_<id>.sh` for base, `serve_<id>_<profile>.sh` otherwise.

    The base name is deliberately unsuffixed: it is what the RUNBOOK, the
    existing scripts and every measurement already taken refer to, and renaming
    it would orphan all of them to no purpose.
    """
    return f"serve_{cfg.id}.sh" if profile == "base" else f"serve_{cfg.id}_{profile}.sh"


def render_serve_script(cfg: ServerConfig, profile: str = "base") -> str:
    prof = get_profile(profile)
    argv = serve_command(cfg, profile)
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

    # Every one of these is omitted in its default state, so the base profile of
    # a config with no KV override renders exactly the script already committed
    # next to its measurements.
    extra_header = ""
    if cfg.kv_cache_dtype != "auto":
        extra_header += f"# KVcache: {cfg.kv_cache_dtype}\n"
    if profile != "base":
        extra_header += f"# Profile: {prof.id} -- {prof.description}\n"
        if not prof.speed_safe:
            extra_header += (
                "#\n"
                "# NOT a speed-measurement server. This profile changes how the\n"
                "# server generates, so a latency or throughput number taken from\n"
                "# it is not comparable with the base profile's. Scoring only.\n"
            )

    return f"""#!/usr/bin/env bash
# Generated by bench.server. Do not edit by hand; edit config/configs.yaml.
#
# Config : {cfg.id}  ({cfg.name})
# Quant  : {cfg.quant_method} / {cfg.bit_width}-bit
# Sparse : {cfg.sparsity_method} / {cfg.sparsity_pattern} @ {cfg.sparsity_ratio}
# Spec   : {cfg.spec_method}
{extra_header}#
# max-model-len and gpu-memory-utilization are pinned to the same values in
# every config so that a smaller checkpoint does not silently receive a larger
# KV cache and get credited for it.
set -euo pipefail

{exports}{body}
"""


def write_serve_scripts(
    configs: dict[str, ServerConfig], out_dir: Path | None = None
) -> list[Path]:
    """One script per (config, profile) the config asks for."""
    out_dir = out_dir or SCRIPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cfg in configs.values():
        for profile in cfg.profiles:
            path = out_dir / script_name(cfg, profile)
            path.write_text(
                render_serve_script(cfg, profile), encoding="utf-8", newline="\n"
            )
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


def probe_capabilities(base_url: str, model: str) -> dict[str, Any]:
    """What the running server can do, as far as the serve profiles differ.

    A server does not report its launch flags, so each profile difference is
    read off behaviour instead:

      tools          a one-token request with `tool_choice: "auto"`. vLLM
                     answers 400 unless it was started with
                     --enable-auto-tool-choice and a parser. None when the
                     probe could not tell either way.
      max_model_len  as /v1/models reports it for `model`, or None.
    """
    base = base_url.rstrip("/")
    caps: dict[str, Any] = {"tools": None, "max_model_len": None}
    try:
        for entry in httpx.get(base + "/v1/models", timeout=10.0).json().get("data", []):
            if entry.get("id") == model and entry.get("max_model_len"):
                caps["max_model_len"] = int(entry["max_model_len"])
    except Exception:  # noqa: BLE001
        pass
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "Does nothing.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    try:
        response = httpx.post(base + "/v1/chat/completions", json=payload, timeout=60.0)
        if response.status_code == 200:
            caps["tools"] = True
        elif response.status_code == 400:
            caps["tools"] = False
    except Exception:  # noqa: BLE001
        pass
    return caps


def profile_problem(profile_id: str, caps: dict[str, Any]) -> str | None:
    """Why a server with `caps` cannot stand in for `profile_id`, or None.

    Checks only what a suite needs from the profile. A `tools` server can
    score a `base` suite -- requests without tools never reach the parser --
    so this is "can it serve", not "is it exactly".
    """
    prof = get_profile(profile_id)
    if "--enable-auto-tool-choice" in prof.extra_args and caps.get("tools") is not True:
        state = "is off" if caps.get("tools") is False else "could not be confirmed"
        return f"native tool calling {state}; start the server from its `{profile_id}` serve script"
    if prof.max_model_len and (caps.get("max_model_len") or 0) < prof.max_model_len:
        return (
            f"server max_model_len is {caps.get('max_model_len')}, the `{profile_id}` "
            f"profile needs {prof.max_model_len}"
        )
    return None


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
