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
