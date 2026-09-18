from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_server() -> str:
    """A mock vLLM on a free port, torn down at the end of the session."""
    import uvicorn

    from mock_server import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(base_url + "/health", timeout=1.0).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    else:
        raise RuntimeError("mock server did not become ready")

    yield base_url

    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="session")
def scored_items():
    """Real per-item results from past runs, paired with the key they were scored against.

    Yields (scorer, key, rows) per items file found under `results/quality/`.
    That directory is gitignored machine state, so this skips on a fresh
    clone rather than failing: it is a regression check against measurements
    this checkout happens to have, not a property of the repository.

    The scorer is taken from `config/suites.yaml` rather than hardcoded, so a
    suite that changes its scorer cannot be silently re-scored under the old one.
    """
    import json

    from bench.classes import REPO_ROOT
    from bench.suites import load_suites

    root = REPO_ROOT / "results" / "quality"
    if not root.is_dir():
        pytest.skip("no results/quality in this checkout")

    suites = load_suites()
    out = []
    for directory in sorted(root.iterdir()):
        suite = suites.get(directory.name)
        if suite is None or not suite.key_file.exists():
            continue
        key = {
            row["scenario_id"]: row
            for row in (
                json.loads(line)
                for line in suite.key_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        for path in sorted(directory.glob("*.jsonl")):
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            # An items file from a run against a different eval build joins by
            # id but answers different questions; skip rather than assert on it.
            if rows and all(r["scenario_id"] in key for r in rows):
                out.append((suite.scorer, key, rows))

    if not out:
        pytest.skip("no scoreable items files under results/quality")
    return out


@pytest.fixture(scope="session")
def ifeval_source():
    """The frozen IFEval items, as (instruction_id_list, kwargs) rows.

    Read from the committed key file rather than fetched, so the verifier sweep
    stays offline like the rest of the run path -- and so it sweeps exactly the
    items this study scores rather than the 541 it drew them from.
    """
    from bench.suites import load_suites, read_key

    suite = load_suites()["ifeval"]
    if not suite.key_file.exists():
        pytest.skip("ifeval eval set has not been built")
    return [row["meta"] for row in read_key(suite).values()]


@pytest.fixture(scope="session")
def bfcl_key():
    """The frozen BFCL key rows, scenario_id -> row. Offline."""
    from bench.suites import load_suites, read_key

    suite = load_suites()["bfcl_ast"]
    if not suite.key_file.exists():
        pytest.skip("bfcl_ast eval set has not been built")
    return read_key(suite)


@pytest.fixture(scope="session")
def bfcl_scenarios():
    """The frozen BFCL prompts. Offline."""
    from bench.scenarios import read_scenarios
    from bench.suites import load_suites

    suite = load_suites()["bfcl_ast"]
    if not suite.prompt_file.exists():
        pytest.skip("bfcl_ast eval set has not been built")
    return read_scenarios(suite.prompt_file)


@pytest.fixture(scope="module")
def configs_fixture():
    """Server configs, for tests outside test_quality's own module fixtures."""
    from bench.server import load_configs

    return load_configs()


@pytest.fixture
def args(tmp_path, monkeypatch):
    """A parsed quality-pass CLI with the items store redirected out of the repo.

    Shared by test_quality and test_tracking: the round-trip tests run the same
    `run_suite` the real pass runs, so they need the same arguments rather than
    a second, drifting copy of them.
    """
    from bench import quality as quality_mod

    monkeypatch.setattr(quality_mod, "QUALITY_ROOT", tmp_path)
    parsed = quality_mod.build_parser().parse_args([])
    parsed.n_items = 8
    parsed.concurrency = 2
    return parsed
