"""One batch run and its result folder: log, settings, predictions, scores, analysis, traces."""
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc

from .config import PROJECT, path
from .data import load_gold
from .evaluation.analysis import analyze, to_markdown

LOG = "log.jsonl"
OLD_LOG = "protokoll.jsonl"   # older agents
RUN_FILES = (LOG, "settings.json", "mapped_predictions.csv", "scores.json", "analysis.json", "analysis.md")
DEFAULT_MAX_STEPS = 14


def condition_name(cfg):
    """Folder name from the conditions of a run, e.g. reasoning-off_kag-on_toc-on_examples-optional."""
    a = cfg["agent"]
    name = (f"reasoning-{'on' if cfg['llm']['reasoning'] else 'off'}_kag-{'on' if a['kag'] else 'off'}"
            f"_toc-{'on' if a['toc'] else 'off'}_examples-{'forced' if a['examples_with_sections'] else 'optional'}")
    if a["examples_gnd"]:
        name += "_examples-gnd"
    if a["max_steps"] != DEFAULT_MAX_STEPS:
        name += f"_steps-{a['max_steps']}"
    return name


def run_folder(cfg, model, name=None):
    """results/<model>/<conditions>, or results/<model>/<name> with an own name."""
    return Path(cfg["run"]["output"]) / model.split("/")[-1] / (name or condition_name(cfg))


def _must_match(settings):
    """What must stay the same within one run, so its documents are comparable."""
    cfg = settings.get("config", {})
    llm, mapping = cfg.get("llm", {}), cfg.get("mapping", {})
    return {"model": settings.get("model"), "conditions": settings.get("conditions"),
            "prompt_hash": settings.get("prompt_hash"),
            "max_tokens": llm.get("max_tokens_reasoning" if llm.get("reasoning") else "max_tokens"),
            **{k: mapping.get(k) for k in ("collection", "alpha", "min_cosine")}}


def prepare_folder(folder, settings, force=False):
    """Makes sure a run never overwrites another one. Returns the settings to store.

    - empty folder: a new run starts
    - same conditions and prompt: the run is resumed, finished documents are kept
    - anything else: stop, unless --force, which deletes the old results of this folder first
    """
    old_file = folder / "settings.json"
    if not old_file.exists() and not (folder / LOG).exists():
        return settings
    if force:
        for name in RUN_FILES:
            (folder / name).unlink(missing_ok=True)
        print(f"--force: old results in {folder} deleted.")
        return settings
    old = json.loads(old_file.read_text(encoding="utf-8")) if old_file.exists() else {}
    stored, current = _must_match(old), _must_match(settings)
    differs = {k: (stored[k], current[k]) for k in current if stored[k] != current[k]}
    if differs:
        lines = "\n".join(f"  {k}: stored {stored}, now {now}" for k, (stored, now) in differs.items())
        sys.exit(f"{folder} already holds another run:\n{lines}\n"
                 "Nothing was changed. Use --name for another folder, or --force to delete the old results.")
    old.setdefault("resumed", []).append(settings["started"])
    old.pop("finished", None)   # running again: set anew at the end
    return old


def load_log(folder, name=LOG):
    """One entry per document; if a document was repeated (e.g. after an error), the last entry counts."""
    file = folder / name
    entries = {}
    if file.exists():
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                entries[e["doc_id"]] = e
    return list(entries.values())


def write_predictions(folder):
    """mapped_predictions.csv in the pipeline format (readable by compute_metrics.R and auswertung.py)."""
    rows = [{"doc_id": e["doc_id"], "label_id": s["gnd_id"], "candidate": s["candidate"], "term": s["term"],
             "cosine_similarity": s["cosine_similarity"], "hybrid_score": s["score"], "score": s["score"]}
            for e in load_log(folder) for s in e.get("suggestions", [])]
    out = folder / "mapped_predictions.csv"
    pd.DataFrame(rows, columns=["doc_id", "label_id", "candidate", "term", "cosine_similarity", "hybrid_score",
                                "score"]).to_csv(out, index=False)
    return out


def write_scores(cfg, folder):
    """scores.json of a run, computed with casimir on the documents of the run. A finished run is never lost
    because of the scoring: if it fails, the reason is printed and it can be repeated with ie-agent score."""
    predictions = write_predictions(folder)
    out = folder / "scores.json"
    out.unlink(missing_ok=True)   # never leave old scores next to new results
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index.arrow"
        table = pa.table({"idn": [e["doc_id"] for e in load_log(folder)]})
        with pa.ipc.new_file(index, table.schema) as writer:
            writer.write_table(table)
        result = subprocess.run(
            [cfg["scoring"]["rscript"], str(PROJECT / "src/ie_agent/evaluation/compute_scores.R"),
             "--predictions", str(predictions), "--index", str(index),
             "--ground-truth", str(path(cfg, "ground_truth")), "--output", str(out),
             "--kind", cfg["scoring"]["kind"], "--n_threads", str(cfg["scoring"]["n_threads"])],
            capture_output=True, text=True)
    if result.returncode or not out.exists():
        name = folder.relative_to(cfg["run"]["output"])
        print(f"Scoring failed, the results are safe. Repeat with: ie-agent score --name {name}\n"
              f"{result.stderr[-1500:]}", flush=True)
        return None
    scores = {"evaluated": _now(), **json.loads(out.read_text(encoding="utf-8"))}
    out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Scores: {out} (F1 {scores['f1@100']['value']:.4f})", flush=True)
    return scores


def write_analysis(cfg, folder):
    """analysis.json and analysis.md of a run: tool usage, decision branches, hits per tool."""
    try:
        result = {"created": _now(), **analyze(load_log(folder), load_gold(cfg))}
        (folder / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                                              encoding="utf-8")
        title = folder.relative_to(cfg["run"]["output"])
        (folder / "analysis.md").write_text(to_markdown(result, title), encoding="utf-8")
        print(f"Analysis: {folder / 'analysis.md'}", flush=True)
    except Exception as e:   # the results of a finished run are safe
        print(f"Analysis failed: {e!r}. Repeat with: ie-agent analyze --name <folder>", flush=True)


def run_batch(cfg, agent, ids, name=None, force=False):
    folder = run_folder(cfg, agent.model, name)
    folder.mkdir(parents=True, exist_ok=True)
    settings = prepare_folder(folder, {
        "model": agent.model, "conditions": condition_name(cfg), "prompt_hash": agent.prompt_hash,
        "git": _git_commit(), "started": _now(), "system_prompt": agent.system_prompt, "config": cfg}, force)
    _write_settings(folder, settings)
    done = {e["doc_id"] for e in load_log(folder) if "error" not in e}   # documents with an error are repeated
    todo = [d for d in ids if d not in done]
    print(f"Model {agent.model} · prompt {agent.prompt_hash} · {condition_name(cfg)}")
    print(f"{len(done)} already done, {len(todo)} to do → {folder}", flush=True)

    with ThreadPoolExecutor(max_workers=cfg["run"]["parallel"]) as pool, open(folder / LOG, "a") as f:
        futures = {pool.submit(agent.annotate, doc): doc for doc in todo}
        try:
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    entry = future.result()
                except Exception as e:   # an error in one document does not stop the run
                    entry = {"doc_id": futures[future], "error": repr(e)}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)}", flush=True)
        except BaseException:   # Ctrl+C: otherwise the pool still works through all waiting documents
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    settings["finished"] = _now()
    _write_settings(folder, settings)
    print(f"Predictions: {write_predictions(folder)}")
    write_scores(cfg, folder)
    write_analysis(cfg, folder)


def write_trace(cfg, agent, title, doc_id):
    """Annotates one document and writes what the model saw, thought and did as Markdown and JSON."""
    r = agent.annotate(doc_id, trace=True)
    out = run_folder(cfg, agent.model, "traces")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{doc_id}_{condition_name(cfg)}"
    (out / f"{stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    md = _trace_markdown(r, title, load_gold(cfg)[doc_id], agent.tools.gnd.label)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(f"Trace: {out / stem}.md (and .json)")


def _trace_markdown(r, title, gold, label):
    t = r["trace"]
    hits = set(r["gnd_ids"]) & gold
    lines = [f"# Reasoning trace: {r['doc_id']}", "",
             f"**Title:** {title}  ", f"**Model:** {t['model']} (reasoning {'on' if t['reasoning'] else 'off'})  ",
             f"**Steps:** {r['n_steps']} · **read:** {', '.join(r['read']) or 'title only'} · **abort:** "
             f"{r['abort'] or '–'} · **tokens in/out:** {r['tokens_in']}/{r['tokens_out']} · {r['seconds']} s", "",
             "| | headings |", "|---|---|",
             f"| assigned | {'; '.join(label(i) for i in r['gnd_ids'])} |",
             f"| gold | {'; '.join(label(i) for i in sorted(gold))} |",
             f"| hits | {len(hits)} of {len(r['gnd_ids'])} assigned, {len(hits)} of {len(gold)} gold |", "",
             "<details><summary>System prompt</summary>", "", "```text", t["system_prompt"], "```", "</details>", "",
             "<details><summary>First message (title and examples)</summary>", "", "```text", t["first_message"],
             "```", "</details>", ""]
    for s in t["steps"]:
        lines += [f"## Step {s['step']}", "",
                  f"Offered: `{'`, `'.join(s['offered'])}`" + (" · **finish forced**" if s["forced"] else ""), ""]
        if s["reasoning"]:
            lines += ["**Reasoning**", "", *[f"> {line}" if line else ">" for line in s["reasoning"].splitlines()], ""]
        if s["content"]:
            lines += ["**Text answer**", "", s["content"], ""]
        for c in s["calls"]:
            lines += [f"**Call:** `{c['tool']}({c['args']})`", ""]
        for res in s["results"]:
            lines += [f"<details><summary>Result of {res['tool']} ({len(res['content'])} characters)</summary>", "",
                      "```text", res["content"], "```", "</details>", ""]
    return "\n".join(lines)


def _write_settings(folder, settings):
    (folder / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
