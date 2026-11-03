"""Starting and stopping the vLLM model server."""
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests


def server_settings(cfg, name):
    """How to start the model <name> from models: in config.yaml (with vllm, log and timeout from server:)."""
    if name not in cfg["models"]:
        known = ", ".join(cfg["models"])
        sys.exit(f"No model '{name}' under models: in config.yaml. Known: {known}")
    return {**cfg["server"], **cfg["models"][name]}


def _answers(url):
    """Does a model server answer on this URL?"""
    try:
        return requests.get(f"{url}/models", timeout=5).ok
    except requests.RequestException:
        return False


def _log_file(s, gpu):
    """results/vllm.log -> results/vllm_gpu<N>.log: one log per GPU, overwritten at every start."""
    log = Path(s["log"])
    log = log.with_name(f"{log.stem}_gpu{gpu}{log.suffix}")
    log.parent.mkdir(parents=True, exist_ok=True)
    return log


def _stop(process):
    """Stops the server together with the engine processes it started (its process group)."""
    if process.poll() is not None:
        # Already stopped.
        return

    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        # Did not stop within a minute: kill it.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextmanager
def vllm_server(s, gpu):
    """Starts a vLLM server on one GPU, waits until it answers and always stops it again (also after an error,
    Ctrl+C or when the terminal is closed)."""
    url = f"http://localhost:{s['port']}/v1"
    if _answers(url):
        sys.exit(f"A server already answers on port {s['port']}. Stop it, or run without --gpu to use it.")

    # 1. Start the server in the background; its output goes to the log.
    log = _log_file(s, gpu)
    command = [
        s["vllm"], "serve", s["name"],
        "--port", str(s["port"]),
        *shlex.split(s["args"]),
    ]
    with open(log, "w") as out:
        # Own process group, so the engine processes vLLM starts are stopped together with it.
        process = subprocess.Popen(
            command,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )

    # Closing tmux sends SIGTERM. Turned into a normal exit, the finally block below still stops the server.
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    start = time.time()
    print(f"Starting {s['name']} on GPU {gpu}, port {s['port']} (log: {log}) ...", flush=True)

    try:
        # 2. Wait until it answers.
        while not _answers(url):
            if process.poll() is not None:
                sys.exit(f"The server stopped while starting, see {log}")
            if time.time() - start > s["timeout"]:
                sys.exit(f"The server did not answer within {s['timeout']} s, see {log}")
            time.sleep(5)

        s["load_seconds"] = round(time.time() - start, 1)  # for resource_stats.json
        print(f"Server ready after {s['load_seconds']:.0f} s.", flush=True)

        # 3. The run uses the server.
        yield url

    finally:
        # 4. Always stop it again.
        signal.signal(signal.SIGTERM, previous)
        _stop(process)
        print(f"Server stopped, GPU {gpu} is free.", flush=True)
