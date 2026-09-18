"""Config loading and serve-script rendering.

The sparsity fields matter here because they are pure metadata: vLLM reads the
pattern out of the checkpoint, so nothing at run time would notice if they were
wrong. Their only job is to reach MLflow, which is what these assert.
"""

from __future__ import annotations

import pytest

from bench.server import (
    PROFILES,
    SCRIPTS_DIR,
    effective_max_model_len,
    load_configs,
    render_serve_script,
    script_name,
    serve_command,
)

SPARSE_CONFIGS = ("sparse24_bf16", "sparse24_w4a16")


@pytest.fixture(scope="module")
def configs():
    return load_configs()


def test_the_sparsity_2x2_is_fully_defined(configs):
    """All four cells must exist or the interaction cannot be computed."""
    cells = {
        (c.sparsity_pattern, c.quant_method)
        for c in configs.values()
        if c.id in ("baseline_bf16", "w4a16_gptq", *SPARSE_CONFIGS)
    }
    assert cells == {("none", "none"), ("none", "gptq"), ("2:4", "none"), ("2:4", "gptq")}


def test_the_quantization_ladder_is_in_the_run_plan(configs):
    for config_id in ("baseline_bf16", "fp8_dynamic", "w4a16_gptq", *SPARSE_CONFIGS):
        assert configs[config_id].enabled, f"{config_id} is not in the run plan"


@pytest.mark.parametrize("config_id", SPARSE_CONFIGS)
def test_sparsity_reaches_mlflow_params(configs, config_id):
    params = configs[config_id].as_params()
    assert params["sparsity_method"] == "sparsegpt"
    assert params["sparsity_pattern"] == "2:4"
    assert params["sparsity_ratio"] == 0.5


def test_dense_configs_declare_no_sparsity(configs):
    for config_id in ("baseline_bf16", "fp8_dynamic", "w4a16_gptq"):
        params = configs[config_id].as_params()
        assert params["sparsity_pattern"] == "none"
        assert params["sparsity_ratio"] == 0.0


def test_serving_flags_are_identical_across_configs(configs):
    """A smaller checkpoint must not receive a larger KV cache and get credit."""
    pinned = {
        config_id: (c.max_model_len, c.gpu_memory_utilization, c.max_num_seqs)
        for config_id, c in configs.items()
    }
    assert len(set(pinned.values())) == 1, pinned


def test_sparsity_needs_no_serve_flags(configs):
    """vLLM reads the pattern from the checkpoint, so nothing is passed on the CLI."""
    argv = " ".join(serve_command(configs["sparse24_w4a16"]))
    assert "spars" not in argv.lower()


def test_server_env_is_exported_and_identical_across_configs(configs):
    """The sampler switch changes kernels, so every config must carry the same env."""
    envs = {config_id: tuple(sorted(c.env.items())) for config_id, c in configs.items()}
    assert len(set(envs.values())) == 1, envs
    for c in configs.values():
        script = render_serve_script(c)
        assert "export VLLM_USE_FLASHINFER_SAMPLER=0\n" in script
        assert script.index("export ") < script.index("vllm serve")
        assert c.as_params()["server_env"] == "VLLM_USE_FLASHINFER_SAMPLER=0"


def test_speculative_config_is_shell_quoted(configs):
    script = render_serve_script(configs["fp8_eagle3"])
    assert "--speculative-config '{" in script


SPEC_CONFIGS = ("bf16_eagle3", "fp8_eagle3")


@pytest.mark.parametrize("config_id", SPEC_CONFIGS)
def test_speculative_configs_match_the_published_model_card(configs, config_id):
    """k=3 and the RedHatAI head, per its card. Not the 5 that reads as default."""
    cfg = configs[config_id]
    assert cfg.spec_method == "eagle3"
    assert cfg.num_speculative_tokens == 3
    assert cfg.draft_model_path == "RedHatAI/Qwen3-8B-speculator.eagle3"


def test_speculative_serve_command_shape(configs):
    """The JSON vLLM expects: method, model, num_speculative_tokens."""
    import json

    argv = serve_command(configs["bf16_eagle3"])
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    assert spec == {
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
    }


def test_speculative_pairs_with_a_matching_non_speculative_control(configs):
    """Each speculative config needs a baseline identical but for the draft.

    Without the pair, a speedup cannot be attributed to speculation rather than
    to the quantization it happens to be stacked on.
    """
    for spec_id, control_id in (("bf16_eagle3", "baseline_bf16"), ("fp8_eagle3", "fp8_dynamic")):
        spec, control = configs[spec_id], configs[control_id]
        assert control.spec_method == "none"
        assert spec.model_path == control.model_path
        assert spec.quant_method == control.quant_method


def test_baseline_configs_pass_no_speculative_flag(configs):
    assert "--speculative-config" not in serve_command(configs["baseline_bf16"])


# --------------------------------------------------------------------------
# The Qwen3-4B DB Bahn SFT family
# --------------------------------------------------------------------------

DBBAHN_CONFIGS = ("dbbahn_bf16", "dbbahn_fp8_eagle3")


def test_dbbahn_family_is_served_under_its_own_name(configs):
    """Not qwen3-8b.

    run.py and quality.py only warn when a config's served_model_name is
    *absent* from /v1/models. Two families sharing one served name would let
    either pass attach to the other's running server and report a complete,
    plausible, mislabelled result -- so the distinct name is what keeps that
    guard working at all.
    """
    for config_id in DBBAHN_CONFIGS:
        assert configs[config_id].served_model_name == "qwen3-4b-dbbahn"

    qwen8b = {c.id for c in configs.values() if c.served_model_name == "qwen3-8b"}
    assert not qwen8b & set(DBBAHN_CONFIGS)


def test_dbbahn_speculator_is_the_4b_head(configs):
    """The 8B head is not merely wrong here, it is architecturally incompatible.

    EAGLE-3 consumes the target's hidden states, so the head's hidden_size must
    match: 2560 for Qwen3-4B against 4096 for Qwen3-8B. RedHat's Qwen3 ladder
    starts at 8B; AngelSlim's covers 4B.
    """
    cfg = configs["dbbahn_fp8_eagle3"]
    assert cfg.spec_method == "eagle3"
    assert cfg.draft_model_path == "AngelSlim/Qwen3-4B_eagle3"
    assert cfg.draft_model_path != configs["fp8_eagle3"].draft_model_path


def test_dbbahn_speculative_config_has_no_control_and_that_is_deliberate(configs):
    """Records a known limitation of the two-config scope, rather than hiding it.

    Every other speculative config in this study is paired with a control
    identical but for the draft model, which is what lets a speedup be
    attributed to speculation rather than to the quantization under it.
    `dbbahn_fp8_eagle3` has no such pair: the chosen scope is before-and-after,
    so its only comparator is `dbbahn_bf16`, which differs by *both*
    quantization and speculation.

    This test passes today and is meant to FAIL the moment a `dbbahn_fp8` entry
    is added -- at which point delete it and add the pair to
    test_speculative_pairs_with_a_matching_non_speculative_control instead.
    """
    controls = [
        c
        for c in configs.values()
        if c.spec_method == "none"
        and c.served_model_name == "qwen3-4b-dbbahn"
        and c.quant_method == configs["dbbahn_fp8_eagle3"].quant_method
    ]
    assert controls == [], (
        "a matching control now exists; the speedup is decomposable. Move this "
        f"pair into the pairing test: {[c.id for c in controls]}"
    )


def test_duplicate_config_id_is_rejected(tmp_path):
    """Silently overwriting would drop a config from the run plan with no error."""
    path = tmp_path / "configs.yaml"
    path.write_text(
        "defaults: {}\n"
        "configs:\n"
        "  - id: x\n"
        "    name: first\n"
        "    model_path: a\n"
        "    served_model_name: x\n"
        "  - id: x\n"
        "    name: second\n"
        "    model_path: b\n"
        "    served_model_name: x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate config id"):
        load_configs(path)


def test_unknown_key_in_a_config_is_rejected(tmp_path):
    path = tmp_path / "configs.yaml"
    path.write_text(
        "defaults: {}\n"
        "configs:\n"
        "  - id: x\n"
        "    name: x\n"
        "    model_path: x\n"
        "    served_model_name: x\n"
        "    sparsity_pattren: '2:4'\n",  # typo
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_configs(path)


# --------------------------------------------------------------------------
# serve profiles
# --------------------------------------------------------------------------


def test_the_base_profile_still_renders_the_committed_scripts(configs):
    """Profiles must not have moved the serving regime already measured under.

    Every number in `mlflow.db` was produced by a server started from one of
    the committed `scripts/serve_*.sh`. If adding profiles changed what those
    files say, the measurements no longer document the server that produced
    them, and re-running a config would not reproduce its own earlier cell.
    """
    for cfg in configs.values():
        path = SCRIPTS_DIR / script_name(cfg, "base")
        if not path.exists():
            continue
        assert path.read_text(encoding="utf-8") == render_serve_script(cfg, "base"), (
            f"{path.name} is no longer what bench.server renders for it"
        )


def test_the_base_profile_adds_nothing(configs):
    """It is the pinned regime, not a variant of it."""
    cfg = configs["baseline_bf16"]
    assert PROFILES["base"].extra_args == ()
    assert PROFILES["base"].max_model_len is None
    assert effective_max_model_len(cfg, "base") == cfg.max_model_len


def test_the_tools_profile_enables_the_parser_under_test(configs):
    """BFCL in native FC mode measures the deployed parser, not a prompt hack."""
    argv = serve_command(configs["baseline_bf16"], "tools")
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"


def test_the_long_profile_reaches_qwen3s_native_window_without_yarn(configs):
    """32768 is native for Qwen3-8B. Anything above it needs static YaRN, which
    Qwen notes degrades short-context quality -- and which would confound rope
    scaling with the KV-cache effect the long arm exists to measure."""
    cfg = configs["baseline_bf16"]
    assert effective_max_model_len(cfg, "long") == 32768
    argv = serve_command(cfg, "long")
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert not any("rope" in a or "yarn" in a.lower() for a in argv)


@pytest.mark.parametrize("profile", ["tools", "long"])
def test_a_profile_that_changes_generation_is_not_speed_safe(profile):
    """The flag the run path keys on.

    Both change how the server generates -- the parser buffers tokens until it
    can name a function, and a different window means a different KV cache
    size -- so a latency number from either is not comparable with the base
    profile's. Marking them is what lets `bench.run` refuse instead of
    quietly producing one.
    """
    assert not PROFILES[profile].speed_safe
    assert "NOT a speed-measurement server" in render_serve_script(
        load_configs()["baseline_bf16"], profile
    )


def test_an_unknown_profile_is_refused_by_name(configs):
    with pytest.raises(KeyError, match="unknown serve profile"):
        serve_command(configs["baseline_bf16"], "turbo")


def test_a_profile_script_is_named_apart_from_the_base_one(configs):
    """Same config, two servers; one filename would overwrite the other."""
    cfg = configs["baseline_bf16"]
    assert script_name(cfg, "base") == "serve_baseline_bf16.sh"
    assert script_name(cfg, "long") == "serve_baseline_bf16_long.sh"


def test_kv_cache_dtype_is_off_the_command_line_until_it_is_set(configs):
    """Absent rather than `--kv-cache-dtype auto`, so the existing scripts are
    untouched and a cell that did not vary the KV cache says nothing about it."""
    argv = serve_command(configs["baseline_bf16"])
    assert "--kv-cache-dtype" not in argv
    assert configs["baseline_bf16"].as_params()["kv_cache_dtype"] == "auto"


def test_every_config_emits_at_least_the_base_profile(configs):
    for cfg in configs.values():
        assert "base" in cfg.profiles, f"{cfg.id} would have no pinned serve script"
        for profile in cfg.profiles:
            assert profile in PROFILES, f"{cfg.id} names unknown profile {profile!r}"
