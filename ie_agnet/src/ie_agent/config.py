"""Settings: config.yaml plus the options of the command line."""
import sys
from pathlib import Path

import yaml

# The ie-agent folder (src/ie_agent/config.py -> two levels up).
PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "config.yaml"


def path(cfg, name):
    """A path from paths: in config.yaml, e.g. path(cfg, "ground_truth")."""
    return Path(cfg["paths"]["root"]) / cfg["paths"][name]


def read_config(file=None):
    """The YAML file. Paths in it are relative to the ie-agent folder, so ie-agent works from any directory."""
    cfg = yaml.safe_load((file or CONFIG).read_text(encoding="utf-8"))
    cfg["run"]["output"] = str(PROJECT / cfg["run"]["output"])
    cfg["server"]["log"] = str(PROJECT / cfg["server"]["log"])
    return cfg


def load_config(args):
    """The YAML file with the options of the command line applied."""
    cfg = read_config(args.config)

    # A path given on the command line is relative to where you are.
    if args.output:
        cfg["run"]["output"] = str(Path(args.output).resolve())

    # Options of the command line override config.yaml. None means "not given": the value from the file stays.
    # Switches (store_true) are False when not given, "or None" turns that into "not given".
    overrides = [
        ("llm", "model", args.model),
        ("llm", "reasoning", args.reasoning or None),
        ("llm", "max_tokens", args.max_tokens),
        ("agent", "max_steps", args.max_steps),
        ("agent", "examples_gnd", args.examples_gnd or None),
        ("agent", "examples_with_sections", args.examples_with_sections or None),
        ("agent", "gnd_context", False if args.no_gnd_context else None),
        ("agent", "toc", False if args.no_toc else None),
        ("agent", "abstract_source", "title_abstract_intro" if args.title_abstract_intro else None),
        ("run", "parallel", args.parallel),
    ]
    for section, key, value in overrides:
        if value is not None:
            cfg[section][key] = value

    # For resource_stats.json: the GPU to measure and how long the model server needed to start.
    cfg["run"].update(
        gpu=getattr(args, "gpu", None),
        load_seconds=getattr(args, "load_seconds", None),
    )

    # An explicit budget applies with and without reasoning.
    if args.max_tokens:
        cfg["llm"]["max_tokens_reasoning"] = args.max_tokens

    # The model server. A model from models: gives the name for the requests and, unless --port is given, the port.
    model = cfg["models"].get(cfg["llm"]["model"])
    port = args.port or (model["port"] if model else None)
    if not port:
        sys.exit(f"Model '{cfg['llm']['model']}' is not under models: in config.yaml. Give its server with --port.")
    if model:
        cfg["llm"]["model"] = model["name"]
    cfg["llm"]["base_url"] = f"http://localhost:{port}/v1"

    return cfg
