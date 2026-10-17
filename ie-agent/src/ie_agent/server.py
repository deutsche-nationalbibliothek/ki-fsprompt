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
        sys.exit(f"No model '{name}' under models: in config.yaml. Known: {', '.join(cfg['models'])}")
    return {**cfg["server"], **cfg["models"][name]}


def _answers(url):
    try:
        return requests.get(f"{url}/models", timeout=5).ok
    except requests.RequestException:
        return False


@contextmanager
def vllm_server(s, gpu):
    """Starts a vLLM server on one GPU, waits until it answers and always stops it again (also after an error,
    Ctrl+C or when the terminal is closed)."""
    url = f"http://localhost:{s['port']}/v1"
    if _answers(url):
        sys.exit(f"A server already answers on port {s['port']}. Stop it, or run without --gpu to use it.")
    log = Path(s["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [s["vllm"], "serve", s["name"], "--port", str(s["port"]), *shlex.split(s["args"])]
    with open(log, "w") as out:
        # own process group, so the engine processes vLLM starts are stopped together with it
        process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # closing tmux: still clean up
    start = time.time()
    print(f"Starting {s['name']} on GPU {gpu}, port {s['port']} (log: {log}) ...", flush=True)
    try:
        while not _answers(url):
            if process.poll() is not None:
                sys.exit(f"The server stopped while starting, see {log}")
            if time.time() - start > s["timeout"]:
                sys.exit(f"The server did not answer within {s['timeout']} s, see {log}")
            time.sleep(5)
        print(f"Server ready after {time.time() - start:.0f} s.", flush=True)
        yield url
    finally:
        signal.signal(signal.SIGTERM, previous)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        print(f"Server stopped, GPU {gpu} is free.", flush=True)
