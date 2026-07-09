"""End-to-end Docker smoke test (pure Python — no bash/curl dependency).

Builds the image and asserts the full service contract over HTTP, mirroring
``scripts/smoke_docker.sh`` (the human-facing convenience) but driven from Python
so it runs unchanged on Windows and Linux CI:

  * WITH the model mounted  -> /health model_loaded=true, /predict 200,
    invalid payload -> 400.
  * WITHOUT the model       -> /health model_loaded=false, /predict 503.

Auto-skips when no Docker daemon is reachable or the model artifact is absent, so
the normal suite stays fast and self-contained.

Run explicitly with:  uv run pytest -m docker
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
IMAGE = "datahow-titer-service"
MODEL = REPO / "artifacts" / "xgb_best.joblib"
PAYLOAD = REPO / "scripts" / "sample_payload.json"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available"),
    pytest.mark.skipif(not MODEL.exists(), reason="xgb_best.joblib not present"),
]


def _run(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)


def _get_health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def _wait_for_health(port: int, expect_loaded: bool, timeout: int = 90) -> dict:
    deadline = time.time() + timeout
    last: dict | None = None
    while time.time() < deadline:
        last = _get_health(port)
        if last is not None and last.get("model_loaded") is expect_loaded:
            return last
        time.sleep(1)
    raise AssertionError(f"health never reached model_loaded={expect_loaded} (last={last})")


def _predict(port: int, payload: dict) -> int:
    """POST /predict and return the HTTP status code."""
    req = urllib.request.Request(
        f"http://localhost:{port}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _run_container(name: str, port: int, *extra: str) -> None:
    _run("docker", "rm", "-f", name, timeout=60)
    result = _run(
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{port}:8000",
        "-e",
        "MODEL_PATH=/app/artifacts/xgb_best.joblib",
        *extra,
        IMAGE,
        timeout=120,
    )
    assert result.returncode == 0, f"docker run failed: {result.stderr}"


def test_docker_service_smoke():
    build = _run("docker", "build", "-t", IMAGE, REPO.as_posix())
    assert build.returncode == 0, f"docker build failed:\n{build.stderr[-2000:]}"

    suffix = uuid.uuid4().hex[:8]
    model_name = f"titer_smoke_model_{suffix}"
    nomodel_name = f"titer_smoke_nomodel_{suffix}"
    port = 8137
    mount = f"{(REPO / 'artifacts').as_posix()}:/app/artifacts:ro"

    # 1. WITH the model mounted.
    try:
        _run_container(model_name, port, "-v", mount)
        _wait_for_health(port, expect_loaded=True)
        payload = json.loads(PAYLOAD.read_text())
        assert _predict(port, payload) == 200
        assert _predict(port, {"timestamps": [0, 0], "values": {}}) == 400  # spec: bad -> 400
    finally:
        _run("docker", "rm", "-f", model_name, timeout=60)

    # 2. WITHOUT the model (no mount) -> starts, but degraded.
    try:
        _run_container(nomodel_name, port)
        _wait_for_health(port, expect_loaded=False)
        assert _predict(port, json.loads(PAYLOAD.read_text())) == 503
    finally:
        _run("docker", "rm", "-f", nomodel_name, timeout=60)
