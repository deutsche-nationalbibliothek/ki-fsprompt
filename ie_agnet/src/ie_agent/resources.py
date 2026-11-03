"""Runtime, tokens and GPU energy of a run (resource_stats.json), as in src/completion.py of KI-FS Prompt.

While documents are annotated, nvidia-smi samples the GPU once per second into gpu_log.csv. nvidia-smi reports the
whole device, so on a shared GPU the values include other processes. Without --gpu (a server that is already
running) the GPU is unknown: then there is no GPU log, and energy, power, utilization and memory stay null.
"""
import json
import subprocess
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

GPU_LOG = "gpu_log.csv"

# Header of gpu_log.csv.
COLUMNS = "timestamp,utilization_gpu_pct,memory_used_mib,power_draw_w"

# The same columns, as nvidia-smi names them.
QUERY = "timestamp,utilization.gpu,memory.used,power.draw"


# ---------------------------------------------------------------------------------------------------------
# Sampling the GPU while a run works
# ---------------------------------------------------------------------------------------------------------

@contextmanager
def gpu_logger(folder, gpu):
    """Samples the GPU while the block runs; appends to gpu_log.csv, so a resumed run keeps its earlier samples."""
    if gpu is None:
        # A server that was already running: the GPU is unknown.
        yield
        return

    file = folder / GPU_LOG
    new = not file.exists()

    with open(file, "a") as out:
        if new:
            out.write(COLUMNS + "\n")
            out.flush()

        # nvidia-smi writes one line per second (-l 1) until it is stopped.
        command = [
            "nvidia-smi",
            "-i", str(gpu),
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
            "-l", "1",
        ]
        try:
            process = subprocess.Popen(command, stdout=out, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("nvidia-smi not found, no GPU log is written", flush=True)
            process = None

        try:
            yield
        finally:
            if process is not None:
                process.terminate()
                process.wait()


def gpu_name(gpu):
    """e.g. "NVIDIA A100 80GB PCIe", or None without a GPU."""
    if gpu is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------------------------------------
# resource_stats.json
# ---------------------------------------------------------------------------------------------------------

def summarize_gpu_log(file):
    """Energy (Wh), mean power and utilization, maximal memory. A pause between two parts of a resumed run does not
    count: a sample stands for at most 5 seconds."""
    try:
        log = pd.read_csv(file, skipinitialspace=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {}

    # Lines nvidia-smi could not fill (e.g. "[N/A]") become NaN; lines without a time are dropped.
    log["timestamp"] = pd.to_datetime(log["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    for column in ["utilization_gpu_pct", "memory_used_mib", "power_draw_w"]:
        log[column] = pd.to_numeric(log[column], errors="coerce")
    log = log.dropna(subset=["timestamp"])
    if log.empty:
        return {}

    # How long each sample stands for: the time until the next one (at most 5 s, the last one 1 s).
    dt = log["timestamp"].diff().shift(-1).dt.total_seconds().fillna(1.0).clip(upper=5.0)

    return {
        "energy_wh_generate": round(float((log["power_draw_w"] * dt).sum()) / 3600, 2),
        "mean_power_w_generate": round(float(log["power_draw_w"].mean()), 1),
        "mean_gpu_util_pct_generate": round(float(log["utilization_gpu_pct"].mean()), 1),
        "max_memory_used_mib": int(log["memory_used_mib"].max()),
        "gpu_log_samples": int(len(log)),
    }


def _duration(session):
    """Seconds between start and end of one session (one start of the run)."""
    started = datetime.fromisoformat(session["started"])
    finished = datetime.fromisoformat(session["finished"])
    return (finished - started).total_seconds()


def write_resource_stats(folder, settings, entries):
    """resource_stats.json from the finished documents (tokens), the sessions in settings.json (time) and
    gpu_log.csv (energy). generate_seconds adds up all sessions of the run, so a resumed run is counted right."""
    entries = [e for e in entries if "error" not in e]
    n = len(entries)
    sessions = settings.get("sessions", [])
    last = sessions[-1] if sessions else {}

    seconds = sum(_duration(s) for s in sessions)
    agent_seconds = sum(e.get("seconds", 0) for e in entries)
    tokens_in = sum(e.get("tokens_in", 0) for e in entries)
    tokens_out = sum(e.get("tokens_out", 0) for e in entries)

    def per_doc(value, digits):
        return round(value / n, digits) if n else None

    stats = {
        "hf_model_name": settings["model"],
        "n_docs": n,
        "load_seconds": last.get("load_seconds"),
        "generate_seconds": round(seconds, 1),
        "seconds_per_doc": per_doc(seconds, 3),
        "agent_seconds_per_doc": per_doc(agent_seconds, 1),
        "parallel_docs": settings["config"]["run"]["parallel"],
        "prompt_tokens": tokens_in,
        "output_tokens": tokens_out,
        "prompt_tokens_per_doc": per_doc(tokens_in, 1),
        "output_tokens_per_doc": per_doc(tokens_out, 1),
        "tokens_per_second": round((tokens_in + tokens_out) / seconds, 1) if seconds else None,
        # GPU values: null without a GPU log, otherwise filled in from gpu_log.csv.
        "energy_wh_generate": None,
        "mean_power_w_generate": None,
        "mean_gpu_util_pct_generate": None,
        "max_memory_used_mib": None,
        "gpu_log_samples": 0,
        **summarize_gpu_log(folder / GPU_LOG),
        "gpu_name": last.get("gpu_name"),
        "cuda_visible_devices": None if last.get("gpu") is None else str(last["gpu"]),
        "sessions": len(sessions),
        "finished_at": settings.get("finished"),
    }

    content = json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    (folder / "resource_stats.json").write_text(content, encoding="utf-8")
    return stats
