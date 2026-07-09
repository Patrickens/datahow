"""Docker command wrapper for Make targets on Windows/MSYS shells."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _docker_executable(name: str) -> str:
    docker = shutil.which(name) or shutil.which("docker.exe") or shutil.which("docker")
    if docker is None:
        raise SystemExit(
            "Docker CLI not found. Install/start Docker Desktop and ensure "
            f"'{name}' is on PATH."
        )
    return docker


def _run(args: list[str]) -> int:
    return subprocess.run(args, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "build", "run"])
    parser.add_argument("--docker", default="docker.exe")
    parser.add_argument("--image", default="datahow-titer-service")
    parser.add_argument("--port", default="9000")
    parser.add_argument("--model-path", default="artifacts/xgb_best.joblib")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    docker = _docker_executable(args.docker)

    if args.command == "check":
        return _run([docker, "--version"])

    if args.command == "build":
        return _run([docker, "build", "-t", args.image, "."])

    artifacts_dir = str(Path(args.artifacts_dir).resolve())
    mounted_model = f"/app/artifacts/{Path(args.model_path).name}"
    return _run(
        [
            docker,
            "run",
            "--rm",
            "-p",
            f"{args.port}:8000",
            "-e",
            f"MODEL_PATH={mounted_model}",
            "--mount",
            f"type=bind,source={artifacts_dir},target=/app/artifacts",
            args.image,
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
