#!/usr/bin/env python3
"""Ray supervisor for the external OPD teacher server.

This keeps the existing ZMQ teacher protocol intact. Ray is only used to place
and hold the teacher GPU allocation; the trainer still talks to the teacher
proxy over TCP.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import ray


TEACHER_ENV_KEYS = {
    "BACKEND",
    "CKPT_PATH",
    "HF_DATASETS_CACHE",
    "HF_HOME",
    "N_LOGPROBS",
    "OPD_PATCH_VLLM_FULL_VOCAB_RAW",
    "PATH",
    "PROXY_BACKEND_PORT",
    "PROXY_FRONTEND_PORT",
    "PYTHON_BIN",
    "PYTHONPATH",
    "STOP_EXISTING_TEACHER_SERVER",
    "TEACHER_ENABLE_PREFIX_CACHING",
    "TEACHER_ENFORCE_EAGER",
    "TEACHER_GPU_MEMORY_UTILIZATION",
    "TEACHER_LOG_DIR",
    "TEACHER_MAX_NUM_BATCHED_TOKENS",
    "TEACHER_MODEL_PATH",
    "TEACHER_REPLICAS",
    "TEACHER_SEQ_LEN",
    "TEACHER_TP_SIZE",
    "TEACHER_WORKER_READY_TIMEOUT",
    "TRANSFORMERS_CACHE",
}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _wait_tcp(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


class TeacherServerSupervisor:
    def __init__(self, env: dict[str, str], script_path: str, log_dir: str):
        self.env = env
        self.script_path = script_path
        self.log_dir = log_dir
        self.started_at = None

    def start(self, host: str, port: int, ready_timeout_s: int, post_start_sleep_s: int) -> dict[str, object]:
        env = os.environ.copy()
        env.update(self.env)
        env.setdefault("PYTHON_BIN", sys.executable)

        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        supervisor_log = log_dir / "ray_supervisor.log"
        script = Path(self.script_path)

        with supervisor_log.open("a", encoding="utf-8") as log:
            log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting teacher via Ray\n")
            log.write(f"script={script}\n")
            log.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")
            log.flush()
            result = subprocess.run(
                ["bash", str(script)],
                cwd=str(script.parent),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            raise RuntimeError(f"teacher start script failed with exit code {result.returncode}; see {supervisor_log}")

        if post_start_sleep_s > 0:
            time.sleep(post_start_sleep_s)

        if not _wait_tcp(host, port, ready_timeout_s):
            raise TimeoutError(f"teacher server was not reachable at {host}:{port} after {ready_timeout_s}s")

        self.started_at = time.time()
        return self.status(host, port)

    def status(self, host: str, port: int) -> dict[str, object]:
        pid_files = {}
        for path in sorted(Path(self.log_dir).glob("*.pid")):
            try:
                pid_files[path.name] = path.read_text(encoding="utf-8").strip()
            except OSError:
                pid_files[path.name] = ""
        return {
            "host": host,
            "port": port,
            "ready": _wait_tcp(host, port, 1),
            "started_at": self.started_at,
            "log_dir": self.log_dir,
            "pid_files": pid_files,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the OPD teacher server inside a Ray actor.")
    parser.add_argument("--actor-name", default=os.environ.get("TEACHER_RAY_ACTOR_NAME", "opd_teacher_server"))
    parser.add_argument("--namespace", default=os.environ.get("TEACHER_RAY_NAMESPACE", "opd"))
    parser.add_argument("--resource-name", default=os.environ.get("TEACHER_RAY_RESOURCE", "teacher"))
    parser.add_argument("--num-cpus", type=float, default=float(os.environ.get("TEACHER_RAY_NUM_CPUS", "1")))
    parser.add_argument("--num-gpus", type=float, default=float(os.environ.get("TEACHER_RAY_NUM_GPUS", "0")))
    parser.add_argument("--ready-timeout", type=int, default=_int_env("TEACHER_RAY_READY_TIMEOUT", 300))
    parser.add_argument("--post-start-sleep", type=int, default=_int_env("TEACHER_RAY_POST_START_SLEEP", 30))
    parser.add_argument("--host", default=os.environ.get("TEACHER_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_int_env("TEACHER_SERVER_PORT", 15555))
    parser.add_argument(
        "--script",
        default=os.environ.get("TEACHER_START_SCRIPT", str(Path(__file__).with_name("start_server.sh"))),
    )
    parser.add_argument("--log-dir", default=os.environ.get("TEACHER_LOG_DIR", str(Path(__file__).parent)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_gpus <= 0:
        replicas = _int_env("TEACHER_REPLICAS", 1)
        tp_size = _int_env("TEACHER_TP_SIZE", 1)
        args.num_gpus = replicas * tp_size

    ray.init(address="auto", namespace=args.namespace)

    env = {key: value for key in TEACHER_ENV_KEYS if (value := os.environ.get(key)) is not None}
    remote_cls = ray.remote(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        resources={args.resource_name: 1e-4},
        max_restarts=0,
    )(TeacherServerSupervisor)

    try:
        old_actor = ray.get_actor(args.actor_name)
    except ValueError:
        old_actor = None
    if old_actor is not None:
        ray.kill(old_actor, no_restart=True)
        time.sleep(2)

    actor = remote_cls.options(name=args.actor_name, lifetime="detached").remote(env, args.script, args.log_dir)
    status = ray.get(actor.start.remote(args.host, args.port, args.ready_timeout, args.post_start_sleep))
    print(f"Ray-managed teacher is ready: {status}", flush=True)


if __name__ == "__main__":
    main()
