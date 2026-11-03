"""Command line: ie-agent run | trace | build-gnd | score | analyze | summary."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import Agent
from .config import PROJECT, load_config, read_config
from .data import Documents, Mapper, load_gold
from .gnd import Gnd, build_gnd
from .evaluation.analysis import score
from .runs import LOG, load_log, run_batch, write_analysis, write_scores, write_trace
from .server import server_settings, vllm_server


# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------

def build_agent(cfg):
    """Loads documents, mapper and GND graph once and builds the agent."""
    docs = Documents(cfg)
    agent = Agent(cfg, docs, Mapper(cfg), Gnd.load(cfg))
    return agent, docs


def read_ids(file):
    """A file with one doc_id per line, looked up where you are, otherwise in the ie-agent folder (eval_ids.txt)."""
    file = Path(file) if Path(file).exists() else PROJECT / file
    return [line.strip() for line in file.read_text().splitlines() if line.strip()]


def select_ids(args, cfg, docs):
    """The documents of a run: --ids, a sample of --n, or all."""
    if args.ids:
        return read_ids(args.ids)
    if args.n is None or args.n >= len(docs.ids):
        return docs.ids
    # A random sample with a fixed seed: the same n documents every time.
    rng = np.random.default_rng(cfg["run"]["seed"])
    return [str(i) for i in rng.choice(docs.ids, args.n, replace=False)]


# ---------------------------------------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------------------------------------

def run(args):
    cfg = load_config(args)
    agent, docs = build_agent(cfg)
    run_batch(cfg, agent, select_ids(args, cfg, docs), args.name, args.force)


def trace(args):
    cfg = load_config(args)
    agent, docs = build_agent(cfg)
    write_trace(cfg, agent, docs.title[args.doc_id], args.doc_id)


def build(args):
    manifest = build_gnd(load_config(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def score_again(args):
    cfg = load_config(args)
    write_scores(cfg, Path(cfg["run"]["output"]) / args.name)


def analyze_again(args):
    cfg = load_config(args)
    write_analysis(cfg, Path(cfg["run"]["output"]) / args.name)


def summary(args):
    """One line per run under the output folder: model, conditions, scores, cost."""
    cfg = load_config(args)
    gold = load_gold(cfg)
    out = Path(cfg["run"]["output"])

    rows = []
    for log in sorted(out.glob(f"*/*/{LOG}")):
        folder = log.parent
        entries = load_log(folder)
        if not entries:
            # E.g. stopped before the first document was done.
            continue

        settings_file = folder / "settings.json"
        settings = json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.exists() else {}
        ids = read_ids(args.ids) if args.ids else [e["doc_id"] for e in entries]
        ok = [e for e in entries if "error" not in e]

        rows.append({
            "folder": str(folder.relative_to(out)),
            "conditions": settings.get("conditions", "?"),
            **score(entries, gold, ids),
            "sec/doc": sum(e["seconds"] for e in ok) / max(1, len(ok)),
            "tokens/doc": sum(e["tokens_in"] for e in ok) / max(1, len(ok)),
        })

    if not rows:
        print(f"No runs in {out}.")
        return

    formatters = {
        "P": "{:.3f}".format,
        "R": "{:.3f}".format,
        "F1": "{:.3f}".format,
        "n": "{:.1f}".format,
        "steps": "{:.1f}".format,
        "aborted": "{:.0f}".format,
        "sec/doc": "{:.0f}".format,
        "tokens/doc": "{:,.0f}".format,
    }
    table = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print(table.to_string(index=False, formatters=formatters))


# ---------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    # allow_abbrev=False: otherwise "run --n 5" is ambiguous with --no-gnd-context and --no-toc.
    p = argparse.ArgumentParser(
        prog="ie-agent",
        description="Agentic subject indexing with GND subject headings",
        allow_abbrev=False,
    )

    # 1. Options for every command (before the command): how the agent works.
    p.add_argument("--config", type=Path, help="own YAML file instead of config.yaml")
    p.add_argument("--model", help="model from models: in config.yaml, e.g. qwen or muse (default: llm.model)")
    p.add_argument("--gpu", type=int, help="start the model on this GPU and stop it at the end")
    p.add_argument("--port", type=int, help="port of an already running vLLM server on localhost")
    p.add_argument("--reasoning", action="store_true", help="enable the reasoning of the model")
    p.add_argument("--max-tokens", type=int, help="output tokens per step, including reasoning")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--examples-gnd", action="store_true", help="variant A: annotate example headings with GND")
    p.add_argument(
        "--examples-with-sections",
        action="store_true",
        help="read_abstract_intro / read_toc always return the examples of their section",
    )
    p.add_argument("--no-gnd-context", action="store_true", help="without gnd_context (no GND graph check)")
    p.add_argument("--no-toc", action="store_true", help="without the table of contents: title, then abstract")
    p.add_argument(
        "--title-abstract-intro",
        action="store_true",
        help="read_abstract_intro returns title + abstract/introduction and its examples (condition C)",
    )
    p.add_argument("--parallel", type=int)
    p.add_argument("--output", help="folder with the runs (default: results)")

    # 2. The commands and their own options (after the command).
    sub = p.add_subparsers(dest="command", required=True)

    def command(name, func, help):
        c = sub.add_parser(name, help=help)
        c.set_defaults(func=func)
        return c

    r = command("run", run, "batch run, stored in results/<model>/<conditions> (resumable: start again)")
    r.add_argument("--name", help="own folder name instead of the conditions")
    r.add_argument("--force", action="store_true", help="delete the old results in the folder and start anew")
    r.add_argument("--n", type=int, help="only a sample of n documents (fixed seed); default: all")
    r.add_argument("--ids", help="file with one doc_id per line, e.g. eval_ids.txt")

    command("build-gnd", build, "download the DNB files and build the GND graph in gnd_graph/")

    t = command("trace", trace, "annotate one document and write its reasoning trace (Markdown and JSON)")
    t.add_argument("--doc-id", required=True)

    folder_help = "path under the output folder, e.g. <model>/<conditions>"
    s = command("score", score_again, "write scores_at_100_R.yaml of a finished run again")
    s.add_argument("--name", required=True, help=folder_help)
    a = command("analyze", analyze_again, "write analysis.md and analysis.json of a finished run again")
    a.add_argument("--name", required=True, help=folder_help)

    m = command("summary", summary, "table of all runs under the output folder")
    m.add_argument("--ids", help="only score these documents, e.g. eval_ids.txt")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Without --gpu (or for commands without a model): just run the command.
    if args.gpu is None or args.func not in (run, trace):
        args.func(args)
        return

    # With --gpu N: start the model on GPU N, run the command, stop the model again.
    cfg = read_config(args.config)
    s = server_settings(cfg, args.model or cfg["llm"]["model"])
    s["port"] += args.gpu  # one server per GPU can run at the same time
    args.port = s["port"]
    with vllm_server(s, args.gpu):
        args.load_seconds = s.get("load_seconds")
        args.func(args)


if __name__ == "__main__":
    main()
