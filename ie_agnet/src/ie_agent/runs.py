"""One batch run and its result folder: log, settings, predictions, scores, analysis, traces."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc
import yaml

from .config import PROJECT, path
from .data import load_gold
from .gnd import Gnd, training_cooccurrence
from .evaluation.analysis import analyze, to_markdown
from .resources import GPU_LOG, gpu_logger, gpu_name, write_resource_stats


LOG = "log.jsonl"

# All files that belong to one run. Used by --force to wipe the folder.
RUN_FILES = (
    LOG,
    "settings.json",
    "mapped_predictions.csv",
    "index.arrow",
    "scores_at_100_R.yaml",
    "analysis.json",
    "analysis.md",
    "resource_stats.json",
    GPU_LOG,
)

# Column order of mapped_predictions.csv.
PREDICTION_COLUMNS = [
    "doc_id",
    "label_id",
    "candidate",
    "term",
    "cosine_similarity",
    "hybrid_score",
    "score",
]

DEFAULT_MAX_STEPS = 14


# ---------------------------------------------------------------------------------------------------------
# The result folder of a run
# ---------------------------------------------------------------------------------------------------------

def _on_off(value):
    return "on" if value else "off"


def condition_name(cfg):
    """Folder name from the conditions of a run, e.g. reasoning-off_gnd_context-on_toc-on_examples-optional."""
    a = cfg["agent"]

    parts = [
        f"reasoning-{_on_off(cfg['llm']['reasoning'])}",
        f"gnd_context-{_on_off(a['gnd_context'])}",
        f"toc-{_on_off(a['toc'])}",
        f"examples-{'forced' if a['examples_with_sections'] else 'optional'}",
    ]

    # Only added when not default, so the usual folder names stay short.
    if a.get("abstract_source", "abstract_intro") == "title_abstract_intro":
        parts.append("bedingung-C")
    if a["examples_gnd"]:
        parts.append("examples-gnd")
    if a["max_steps"] != DEFAULT_MAX_STEPS:
        parts.append(f"steps-{a['max_steps']}")

    return "_".join(parts)


def run_folder(cfg, model, name=None):
    """results/<model>/<conditions>, or results/<model>/<name> with an own name."""
    # Qwen/Qwen3.8-27B-FP8 -> Qwen3.8-27B-FP8
    model_name = model.split("/")[-1]
    subfolder = name or condition_name(cfg)
    return Path(cfg["run"]["output"]) / model_name / subfolder


def _must_match(settings):
    """What must stay the same within one run, so its documents are comparable."""
    cfg = settings.get("config", {})
    llm = cfg.get("llm", {})
    mapping = cfg.get("mapping", {})

    # Reasoning changes which token limit applies.
    max_tokens_key = "max_tokens_reasoning" if llm.get("reasoning") else "max_tokens"

    return {
        "model": settings.get("model"),
        "conditions": settings.get("conditions"),
        "prompt_hash": settings.get("prompt_hash"),
        "max_tokens": llm.get(max_tokens_key),
        "collection": mapping.get("collection"),
        "alpha": mapping.get("alpha"),
        "min_cosine": mapping.get("min_cosine"),
    }


def prepare_folder(folder, settings, force=False):
    """Makes sure a run never overwrites another one. Returns the settings to store.

    - empty folder: a new run starts
    - same conditions and prompt: the run is resumed, finished documents are kept
    - anything else: stop, unless --force, which deletes the old results of this folder first
    """
    settings_file = folder / "settings.json"
    log_file = folder / LOG

    # Case 1: empty folder -> start a new run.
    if not settings_file.exists() and not log_file.exists():
        return settings

    # Case 2: --force -> delete everything old in this folder.
    if force:
        for name in RUN_FILES:
            (folder / name).unlink(missing_ok=True)
        print(f"--force: old results in {folder} deleted.")
        return settings

    # Case 3: compare the old settings with the current ones.
    old = (
        json.loads(settings_file.read_text(encoding="utf-8"))
        if settings_file.exists()
        else {}
    )
    stored = _must_match(old)
    current = _must_match(settings)

    differences = [
        f"  {key}: stored {stored[key]}, now {current[key]}"
        for key in current
        if stored[key] != current[key]
    ]

    if differences:
        sys.exit(
            f"{folder} already holds another run:\n"
            + "\n".join(differences)
            + "\nNothing was changed. Use --name for another folder, "
              "or --force to delete the old results."
        )

    # Case 4: same run -> resume it.
    old.setdefault("resumed", []).append(settings["started"])
    old.pop("finished", None)  # set anew at the end
    return old


def load_log(folder):
    """One entry per document; if a document was repeated (e.g. after an error), the last entry counts."""
    file = folder / LOG
    if not file.exists():
        return []

    entries = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        # A later line of the same document replaces the earlier one.
        entries[entry["doc_id"]] = entry

    return list(entries.values())


def _write_settings(folder, settings):
    content = json.dumps(settings, ensure_ascii=False, indent=2)
    (folder / "settings.json").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------
# Evaluation of a finished run: predictions, scores (R), analysis
# ---------------------------------------------------------------------------------------------------------

def write_predictions(folder):
    """mapped_predictions.csv in the pipeline format (as src/mapping.py of KI-FS Prompt): one row per heading."""
    rows = []
    for entry in load_log(folder):
        for s in entry.get("suggestions", []):
            rows.append(
                {
                    "doc_id": entry["doc_id"],
                    "label_id": s["gnd_id"],
                    "candidate": s["candidate"],
                    "term": s["term"],
                    "cosine_similarity": s["cosine_similarity"],
                    "hybrid_score": s["score"],
                    "score": s["score"],  # the pipeline format has both
                }
            )

    out = folder / "mapped_predictions.csv"
    # Columns: header even without rows.
    pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(out, index=False)
    return out


def write_index(folder):
    """index.arrow: the documents of the run (column idn). R takes the gold of exactly these documents, so a document
    without any heading counts as 0 instead of being left out. In KI-FS Prompt this is the fixed validate.arrow."""
    out = folder / "index.arrow"
    table = pa.table({"idn": [e["doc_id"] for e in load_log(folder)]})

    with pa.ipc.new_file(out, table.schema) as writer:
        writer.write_table(table)

    return out


def write_scores(cfg, folder):
    """scores_at_100_R.yaml of a run, computed with casimir by compute_scores.R (in KI-FS Prompt: the stage
    compute_metrics_modelwise in dvc.yaml). A finished run is never lost because of the scoring: if R fails, the
    reason is printed and it can be repeated with ie-agent score."""
    out = folder / "scores_at_100_R.yaml"
    out.unlink(missing_ok=True)  # never leave old scores next to new results

    command = [
        cfg["scoring"]["rscript"],
        str(PROJECT / "src/ie_agent/evaluation/compute_scores.R"),
        "--predictions", str(write_predictions(folder)),
        "--index", str(write_index(folder)),
        "--ground-truth", str(path(cfg, "ground_truth")),
        "--output", str(out),
        "--kind", cfg["scoring"]["kind"],
        "--n_threads", str(cfg["scoring"]["n_threads"]),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0 or not out.exists():
        name = folder.relative_to(cfg["run"]["output"])
        print(
            f"Scoring failed, the results are safe. "
            f"Repeat with: ie-agent score --name {name}\n"
            f"{result.stderr[-1500:]}",
            flush=True,
        )
        return None

    scores = yaml.safe_load(out.read_text(encoding="utf-8"))
    print(f"Scores: {out} (F1 {scores['f1@100']['value']:.4f})", flush=True)
    return scores


def expansions(cfg):
    """Candidate expansions whose reach the analysis measures: GND neighbours and joint assignment in training."""
    cooccurring = training_cooccurrence(cfg)

    result = {
        "co-assigned in training (top 10)": lambda i: cooccurring.get(i, []),
    }

    try:
        gnd = Gnd.load(cfg)
        result["GND neighbours (broader, narrower, related)"] = gnd.neighbours
    except FileNotFoundError:
        # No gnd_graph/ built: measure the co-assignment only.
        pass

    return result


def write_analysis(cfg, folder):
    """analysis.json and analysis.md of a run: tool usage, decision branches, hits per tool."""
    try:
        result = {
            "created": _now(),
            **analyze(load_log(folder), load_gold(cfg), expansions(cfg)),
        }

        analysis_json = json.dumps(
            result, ensure_ascii=False, indent=2, default=float
        )
        (folder / "analysis.json").write_text(analysis_json, encoding="utf-8")

        title = folder.relative_to(cfg["run"]["output"])
        (folder / "analysis.md").write_text(
            to_markdown(result, title), encoding="utf-8"
        )

        print(f"Analysis: {folder / 'analysis.md'}", flush=True)

    except Exception as e:
        # The results of a finished run are safe.
        print(
            f"Analysis failed: {e!r}. "
            f"Repeat with: ie-agent analyze --name <folder>",
            flush=True,
        )


# ---------------------------------------------------------------------------------------------------------
# One batch run
# ---------------------------------------------------------------------------------------------------------

def run_batch(cfg, agent, ids, name=None, force=False):
    """Annotates the documents that are not done yet, then writes predictions, scores and analysis."""
    # 1. The folder: new, resumed, or refused (see prepare_folder).
    folder = run_folder(cfg, agent.model, name)
    folder.mkdir(parents=True, exist_ok=True)

    settings = prepare_folder(
        folder,
        {
            "model": agent.model,
            "conditions": condition_name(cfg),
            "prompt_hash": agent.prompt_hash,
            "git": _git_commit(),
            "started": _now(),
            "system_prompt": agent.system_prompt,
            "config": cfg,
        },
        force,
    )
    _write_settings(folder, settings)

    # 2. What is left: documents with an error are repeated.
    done = {e["doc_id"] for e in load_log(folder) if "error" not in e}
    todo = [d for d in ids if d not in done]

    print(f"Model {agent.model} · prompt {agent.prompt_hash} · {condition_name(cfg)}")
    print(f"{len(done)} already done, {len(todo)} to do → {folder}", flush=True)

    # 3. Annotate, while nvidia-smi samples the GPU.
    gpu = cfg["run"].get("gpu")
    session = {"started": _now()}

    with gpu_logger(folder, gpu):
        _annotate_all(agent, todo, folder / LOG, cfg["run"]["parallel"])

    # 4. Bookkeeping: one session per start of the run, for resource_stats.json.
    settings["finished"] = session["finished"] = _now()
    session.update(
        gpu=gpu,
        gpu_name=gpu_name(gpu),
        load_seconds=cfg["run"].get("load_seconds"),
    )
    settings.setdefault("sessions", []).append(session)
    _write_settings(folder, settings)
    write_resource_stats(folder, settings, load_log(folder))

    # 5. Evaluation
    print(f"Predictions: {write_predictions(folder)}")
    write_scores(cfg, folder)
    write_analysis(cfg, folder)


def _annotate_all(agent, todo, log_file, parallel):
    """Annotates the documents in parallel threads. Each result is appended to the log as soon as it is done, so an
    interrupted run loses only the documents that were still running."""
    with ThreadPoolExecutor(max_workers=parallel) as pool, open(log_file, "a") as log:
        futures = {pool.submit(agent.annotate, doc): doc for doc in todo}

        try:
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    entry = future.result()
                except Exception as e:
                    # An error in one document does not stop the run.
                    entry = {"doc_id": futures[future], "error": repr(e)}

                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log.flush()

                if i % 10 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)}", flush=True)

        except BaseException:
            # Ctrl+C: otherwise the pool still works through all waiting documents.
            pool.shutdown(wait=False, cancel_futures=True)
            raise


# ---------------------------------------------------------------------------------------------------------
# Trace of one document
# ---------------------------------------------------------------------------------------------------------

def write_trace(cfg, agent, title, doc_id):
    """Annotates one document and writes what the model saw, thought and did as Markdown and JSON."""
    result = agent.annotate(doc_id, trace=True)

    out = run_folder(cfg, agent.model, "traces")
    out.mkdir(parents=True, exist_ok=True)

    stem = f"{doc_id}_{condition_name(cfg)}"

    (out / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    markdown = _trace_markdown(
        result, title, load_gold(cfg)[doc_id], agent.tools.gnd.label
    )
    (out / f"{stem}.md").write_text(markdown, encoding="utf-8")

    print(f"Trace: {out / stem}.md (and .json)")


def _trace_markdown(r, title, gold, label):
    """The trace as Markdown: summary and comparison with the gold first, then every step."""
    t = r["trace"]
    hits = set(r["gnd_ids"]) & gold
    read = ", ".join(r["read"]) or "title only"

    lines = [
        f"# Reasoning trace: {r['doc_id']}",
        "",
        f"**Title:** {title}  ",
        f"**Model:** {t['model']} (reasoning {_on_off(t['reasoning'])})  ",
        f"**Steps:** {r['n_steps']} · "
        f"**read:** {read} · "
        f"**abort:** {r['abort'] or '–'} · "
        f"**tokens in/out:** {r['tokens_in']}/{r['tokens_out']} · "
        f"{r['seconds']} s",
        "",
        "| | headings |",
        "|---|---|",
        f"| assigned | {'; '.join(label(i) for i in r['gnd_ids'])} |",
        f"| gold | {'; '.join(label(i) for i in sorted(gold))} |",
        f"| hits | {len(hits)} of {len(r['gnd_ids'])} assigned, "
        f"{len(hits)} of {len(gold)} gold |",
        "",
        *_folded("System prompt", t["system_prompt"]),
        *_folded("First message (title and examples)", t["first_message"]),
    ]

    for step in t["steps"]:
        lines += _trace_step(step)

    return "\n".join(lines)


def _trace_step(s):
    """One request to the model: offered tools, reasoning, text, tool calls and their results."""
    offered = "`, `".join(s["offered"])
    forced = " · **finish forced**" if s["forced"] else ""

    lines = [
        f"## Step {s['step']}",
        "",
        f"Offered: `{offered}`{forced}",
        "",
    ]

    if s["reasoning"]:
        quoted = [
            f"> {line}" if line else ">"
            for line in s["reasoning"].splitlines()
        ]
        lines += ["**Reasoning**", "", *quoted, ""]

    if s["content"]:
        lines += ["**Text answer**", "", s["content"], ""]

    for call in s["calls"]:
        lines += [f"**Call:** `{call['tool']}({call['args']})`", ""]

    for result in s["results"]:
        lines += _folded(
            f"Result of {result['tool']} ({len(result['content'])} characters)",
            result["content"],
        )

    return lines


def _folded(summary, text):
    """A block that is folded in the Markdown view and opens on click."""
    return [
        f"<details><summary>{summary}</summary>",
        "",
        "```text",
        text,
        "```",
        "</details>",
        "",
    ]


# ---------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------

def _now():
    return datetime.now().isoformat(timespec="seconds")


def _git_commit():
    """Short hash of the current commit, stored with every run (None outside git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None