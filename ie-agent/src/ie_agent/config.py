"""Settings: config.yaml plus the options of the command line."""
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]   # the ie-agent folder
CONFIG = PROJECT / "config.yaml"


def path(cfg, name):
    return Path(cfg["paths"]["root"]) / cfg["paths"][name]


def read_config(file=None):
    """The YAML file; paths in it are relative to the ie-agent folder, so ie-agent works from any directory."""
    cfg = yaml.safe_load((file or CONFIG).read_text(encoding="utf-8"))
    cfg["run"]["output"] = str(PROJECT / cfg["run"]["output"])
    cfg["server"]["log"] = str(PROJECT / cfg["server"]["log"])
    return cfg


def load_config(args):
    """The YAML file with the options of the command line applied."""
    cfg = read_config(args.config)
    if args.output:   # a path given on the command line is relative to where you are
        cfg["run"]["output"] = str(Path(args.output).resolve())
    for section, key, value in [("llm", "model", args.model), ("llm", "reasoning", args.reasoning or None),
                                ("llm", "max_tokens", args.max_tokens), ("agent", "max_steps", args.max_steps),
                                ("agent", "examples_gnd", args.examples_gnd or None),
                                ("agent", "examples_with_sections", args.examples_with_sections or None),
                                ("agent", "kag", False if args.no_kag else None),
                                ("agent", "toc", False if args.no_toc else None),
                                ("run", "parallel", args.parallel)]:
        if value is not None:
            cfg[section][key] = value
    if args.max_tokens:   # an explicit budget applies with and without reasoning
        cfg["llm"]["max_tokens_reasoning"] = args.max_tokens

    # a model from models: gives the name for the requests and, unless --port is given, the port
    model = cfg["models"].get(cfg["llm"]["model"])
    port = args.port or (model["port"] if model else None)
    if not port:
        sys.exit(f"Model '{cfg['llm']['model']}' is not under models: in config.yaml. Give its server with --port.")
    if model:
        cfg["llm"]["model"] = model["name"]
    cfg["llm"]["base_url"] = f"http://localhost:{port}/v1"
    return cfg
