"""Config loading and serve-script rendering.

The sparsity fields matter here because they are pure metadata: vLLM reads the
pattern out of the checkpoint, so nothing at run time would notice if they were
wrong. Their only job is to reach MLflow, which is what these assert.
"""

from __future__ import annotations

import pytest

from bench.server import load_configs, render_serve_script, serve_command

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
