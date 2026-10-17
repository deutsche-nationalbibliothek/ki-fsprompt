"""Command line: ie-agent run | sweep | one | trace | build-gnd | score | analyze | summary | eval."""
import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import Agent
from .config import PROJECT, load_config, read_config
from .data import Documents, Gnd, Mapper, build_gnd, load_gold
from .evaluation.analysis import score
from .runs import LOG, OLD_LOG, condition_name, load_log, run_batch, write_analysis, write_scores, write_trace
from .server import server_settings, vllm_server


def build_agent(cfg):
    docs = Documents(cfg)
    return Agent(cfg, docs, Mapper(cfg), Gnd.load(cfg)), docs


def read_ids(file):
    """A file with one doc_id per line, looked up where you are, otherwise in the ie-agent folder (eval_ids.txt)."""
    file = Path(file) if Path(file).exists() else PROJECT / file
    return [line.strip() for line in file.read_text().splitlines() if line.strip()]


def select_ids(args, cfg, docs):
    if args.ids:
        return read_ids(args.ids)
    if args.n is None or args.n >= len(docs.ids):
        return docs.ids   # default: all documents
    return [str(i) for i in np.random.default_rng(cfg["run"]["seed"]).choice(docs.ids, args.n, replace=False)]


def run(args):
    cfg = load_config(args)
    agent, docs = build_agent(cfg)
    run_batch(cfg, agent, select_ids(args, cfg, docs), args.name, args.force)


def sweep(args):
    """One run per combination of the values under sweep: in config.yaml. Data, GND graph and mapper are loaded once,
    finished combinations are skipped like any resumed run."""
    base = load_config(args)
    docs, mapper, gnd = Documents(base), Mapper(base), Gnd.load(base)
    ids = select_ids(args, base, docs)
    grid = base["sweep"]
    combinations = list(itertools.product(grid["reasoning"], grid["kag"], grid["examples_with_sections"]))
    for i, (reasoning, kag, examples) in enumerate(combinations, 1):
        cfg = copy.deepcopy(base)
        cfg["llm"]["reasoning"] = reasoning
        cfg["agent"].update(kag=kag, examples_with_sections=examples)
        print(f"\n=== {i}/{len(combinations)} {condition_name(cfg)} ===", flush=True)
        run_batch(cfg, Agent(cfg, docs, mapper, gnd), ids, force=args.force)
    print()
    summary(args)


def one(args):
    agent, _ = build_agent(load_config(args))
    print(json.dumps(agent.annotate(args.doc_id), ensure_ascii=False, indent=2))


def trace(args):
    cfg = load_config(args)
    agent, docs = build_agent(cfg)
    write_trace(cfg, agent, docs.title[args.doc_id], args.doc_id)


def build(args):
    print(json.dumps(build_gnd(load_config(args)), ensure_ascii=False, indent=2))


def score_again(args):
    cfg = load_config(args)
    write_scores(cfg, Path(cfg["run"]["output"]) / args.name)


def analyze_again(args):
    cfg = load_config(args)
    write_analysis(cfg, Path(cfg["run"]["output"]) / args.name)


def summary(args):
    """One line per run under the output folder: model, conditions, scores, cost."""
    cfg = load_config(args)
    gold, out = load_gold(cfg), Path(cfg["run"]["output"])
    rows = []
    for log in sorted(out.glob(f"*/*/{LOG}")):
        folder = log.parent
        settings_file = folder / "settings.json"
        settings = json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.exists() else {}
        entries = load_log(folder)
        if not entries:   # e.g. stopped before the first document was done
            continue
        ids = read_ids(args.ids) if args.ids else [e["doc_id"] for e in entries]
        ok =[e for e in entries if "error" not in e]
        rows.append({"folder": str(folder.relative_to(out)), "conditions": settings.get("conditions", "?"),
                     **score(entries, gold, ids),
                     "sec/doc": sum(e["seconds"] for e in ok) / max(1, len(ok)),
                     "tokens/doc": sum(e["tokens_in"] for e in ok) / max(1, len(ok))})
    if not rows:
        print(f"No runs in {out}.")
        return
    t = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print(t.to_string(index=False, formatters={c: "{:.3f}".format for c in ("P", "R", "F1")} | {
        "n": "{:.1f}".format, "steps": "{:.1f}".format, "aborted": "{:.0f}".format, "sec/doc": "{:.0f}".format,
        "tokens/doc": "{:,.0f}".format}))


def evaluate(args):
    cfg = load_config(args)
    gold, out = load_gold(cfg), Path(cfg["run"]["output"])

    def entries_of(folder):   # also reads runs of the older agents
        return load_log(folder) or load_log(folder, OLD_LOG)

    entries = entries_of(out / args.name)
    ids = read_ids(args.ids) if args.ids else [e["doc_id"] for e in entries]
    results = {args.name: score(entries, gold, ids)}
    if args.compare:
        results[args.compare] = score(entries_of(out / args.compare), gold, ids)
    cols = ["P", "R", "F1", "n", "steps", "aborted", "docs"]
    print(f"{'run':<32}" + "".join(f"{c:>9}" for c in cols))
    for name, r in results.items():
        print(f"{name:<32}" + "".join(f"{r[c]:>9.3f}" if c in ("P", "R", "F1") else f"{r[c]:>9.1f}" for c in cols))
    if args.compare:
        delta = results[args.name]["F1"] - results[args.compare]["F1"]
        print(f"ΔF1 = {delta:+.4f}")
        if delta < -args.tolerance:
            print(f"F1 dropped by more than {args.tolerance}.")
            sys.exit(1)


def parse_args(argv=None):
    # allow_abbrev=False: otherwise "run --n 5" is ambiguous with --no-kag and --no-toc
    p = argparse.ArgumentParser(prog="ie-agent", description="Agentic subject indexing with GND subject headings",
                                allow_abbrev=False)
    p.add_argument("--config", type=Path, help="own YAML file instead of config.yaml")
    p.add_argument("--model", help="model from models: in config.yaml, e.g. qwen or muse (default: llm.model)")
    p.add_argument("--gpu", type=int, help="start the model on this GPU and stop it at the end")
    p.add_argument("--port", type=int, help="port of an already running vLLM server on localhost")
    p.add_argument("--reasoning", action="store_true", help="enable the reasoning of the model")
    p.add_argument("--max-tokens", type=int, help="output tokens per step, including reasoning")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--examples-gnd", action="store_true", help="variant A: annotate example headings with GND")
    p.add_argument("--examples-with-sections", action="store_true",
                   help="read_abstract_intro / read_toc always return the examples of their section")
    p.add_argument("--no-kag", action="store_true", help="without gnd_context (no GND graph check)")
    p.add_argument("--no-toc", action="store_true", help="without the table of contents: title, then abstract")
    p.add_argument("--parallel", type=int)
    p.add_argument("--output", help="folder with the runs (default: results)")
    sub = p.add_subparsers(dest="command", required=True)

    def command(name, func, help):
        c = sub.add_parser(name, help=help)
        c.set_defaults(func=func)
        return c

    def documents(c):
        c.add_argument("--n", type=int, help="only a sample of n documents (fixed seed); default: all")
        c.add_argument("--ids", help="file with one doc_id per line, e.g. eval_ids.txt")

    r = command("run", run, "batch run, stored in results/<model>/<conditions> (resumable: start again)")
    r.add_argument("--name", help="own folder name instead of the conditions")
    r.add_argument("--force", action="store_true", help="delete the old results in the folder and start anew")
    documents(r)

    w = command("sweep", sweep, "one run per combination of the conditions under sweep: in config.yaml")
    w.add_argument("--force", action="store_true", help="delete old results of these conditions and start anew")
    documents(w)

    command("build-gnd", build, "download the DNB files and build the GND graph in gnd_graph/")
    command("one", one, "annotate one document and print the result").add_argument("--doc-id", required=True)
    command("trace", trace, "annotate one document and write its reasoning trace (Markdown and JSON)").add_argument(
        "--doc-id", required=True)

    for name, func, text in [("score", score_again, "write scores.json of a finished run again"),
                             ("analyze", analyze_again, "write analysis.md and analysis.json of a finished run again")]:
        command(name, func, text).add_argument("--name", required=True,
                                               help="path under the output folder, e.g. <model>/<conditions>")

    command("summary", summary, "table of all runs under the output folder").add_argument(
        "--ids", help="only score these documents, e.g. eval_ids.txt")

    e = command("eval", evaluate, "score a run against the gold, optionally compare with an earlier run")
    e.add_argument("--name", required=True, help="path under the output folder, e.g. <model>/<conditions>")
    e.add_argument("--compare", help="path of an earlier run")
    e.add_argument("--ids", help="only score these documents, e.g. eval_ids.txt")
    e.add_argument("--tolerance", type=float, default=0.01, help="allowed F1 loss, otherwise exit code 1")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is None or args.func not in (run, sweep, one, trace):
        args.func(args)
        return
    cfg = read_config(args.config)
    s = server_settings(cfg, args.model or cfg["llm"]["model"])
    args.port = s["port"]
    with vllm_server(s, args.gpu):
        args.func(args)


if __name__ == "__main__":
    main()
