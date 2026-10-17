# ie-agent

Agentic subject indexing with GND subject headings. A language model assigns headings the way human indexers work
(from the interviews): title first, then abstract or introduction if needed, then map candidates onto the GND, then
the table of contents if still needed. Before finishing it checks its candidates in the GND graph (KAG).

## Quick start

**Once, on a new machine or after cloning:**

```bash
cd ~/ki-fsprompt/ie-agent
uv sync                          # installs the environment of this project (for tests and notebooks)
uv tool install --editable .     # installs the command ie-agent, usable from any folder
ie-agent build-gnd               # downloads and builds the GND knowledge (about 40 s)
```

`uv tool install` puts `ie-agent` into `~/.local/bin`, which is on the `PATH` (open a new terminal if the command
is not found yet). Because of `--editable`, changes to the code take effect at once. Only when the dependencies in
`pyproject.toml` change, repeat `uv tool install --editable .` once.

**Every time:**

```bash
ie-agent --gpu 1 run             # one complete run on GPU 1
```

This starts the language model on GPU 1, waits until it is ready, annotates all documents, writes the results to
`ie-agent/results/` and stops the model again. Run it inside `tmux`, then you can close the terminal (`Ctrl+b`, then
`d`) and come back later with `tmux attach`.

`ie-agent` works from any folder: paths in `config.yaml` (results, GND data, `eval_ids.txt`) are relative to the
`ie-agent` folder. Without the installation, `ie-agent …` inside the `ie-agent` folder does the same.

**Needed beforehand:** Weaviate (port 8087) and Text Embeddings Inference (port 8090) must be running; they come
from the main project (`docker-compose.yaml` there). The GPU you name must be free.

## What a run does by default

`ie-agent --gpu 1 run` without further options uses the values in `config.yaml`:

| setting | default | change with |
|---|---|---|
| model | Qwen3.8-27B-FP8 (`qwen` under `models:`) | `--model muse` |
| where the model runs | started on the GPU you name, port 9514, stopped at the end | `--gpu N`, or `--port` for a running server |
| documents | all 827 documents of the validate split | `--ids eval_ids.txt` (100 fixed ones), `--n 50` (random sample) |
| reasoning | off | `--reasoning` |
| KAG (`gnd_context`, check in the GND graph) | on | `--no-kag` |
| table of contents (`read_toc`) | available | `--no-toc`: title, then abstract or introduction |
| examples for abstract and table of contents | optional: the model asks for them with `get_examples` | `--examples-with-sections`: always delivered with each section |
| steps per document | at most 14, `finish` is forced in the last one | `--max-steps N` |
| output tokens per step | 2048 (8192 with reasoning) | `--max-tokens N` |
| documents in parallel | 8 | `--parallel N` |
| results | `results/Qwen3.8-27B-FP8/reasoning-off_kag-on_toc-on_examples-optional/` | follows the conditions; `--name X` for an own folder name |

These defaults are the conditions of Agent IE v4 (`Cezars_Notitzen/agent_ie.py`), with the corrected GND knowledge.

Options that change the run (`--model`, `--gpu`, `--reasoning`, …) go **before** the command, options of the
command (`--ids`, `--force`) **after** it:

```bash
ie-agent --gpu 1 --reasoning --no-kag run --ids eval_ids.txt
#        └──── how the agent works ─────┘     └ which documents ┘
```

## Typical tasks

```bash
# a run with other conditions: it finds its own folder
ie-agent --gpu 1 --reasoning run
ie-agent --gpu 1 --examples-with-sections run --ids eval_ids.txt

# all 8 combinations of reasoning × KAG × examples, the model is started only once
ie-agent --gpu 1 sweep --ids eval_ids.txt

# without the table of contents
ie-agent --gpu 1 --no-toc run

# another model (defined under models: in config.yaml)
ie-agent --gpu 1 --model muse run

# use a model server that is already running (no --gpu)
ie-agent --port 9514 run

# look at one document: result, or a readable trace of every step with the model's reasoning
ie-agent --gpu 1 one --doc-id 1007748524
ie-agent --gpu 1 --reasoning trace --doc-id 1007748524
```

A run that was interrupted (error, `Ctrl+C`, power cut) is continued by starting the same command again: finished
documents are kept, only the missing ones run.

## Watching a run and reading the results

```bash
wc -l ~/ki-fsprompt/ie-agent/results/Qwen3.8-27B-FP8/*/log.jsonl # documents done so far, per run
tail -f ~/ki-fsprompt/ie-agent/results/vllm.log                  # what the model server is doing
ie-agent summary                        # table of all runs: P, R, F1, headings, steps, aborts, time, tokens
ie-agent summary --ids eval_ids.txt     # the same, all runs on the same 100 documents
ie-agent eval --name Qwen3.8-27B-FP8/reasoning-off_kag-on_toc-off_examples-optional \
              --compare Qwen3.8-27B-FP8/reasoning-off_kag-on_toc-on_examples-optional
ie-agent --output ~/ki-fsprompt/Cezars_Notitzen/agent_runs eval --name qwen_agent_ie_v4_voll   # older runs
```

`eval` scores against the title gold (set precision, recall and F1 per document, averaged, as in
`Cezars_Notitzen/auswertung.py`); documents without a result count as 0. With `--compare` it exits with code 1 if
F1 dropped by more than `--tolerance` (default 0.01). Compare runs only on the same documents: two runs on
different documents can differ by several points by chance.

### Where results go

```
results/
  Qwen3.8-27B-FP8/
    reasoning-off_kag-on_toc-on_examples-optional/    one folder per run, named after its conditions
    reasoning-off_kag-on_toc-off_examples-optional/
    traces/                     reasoning traces (ie-agent trace)
  vllm.log                      output of the last model server
```

| file | content |
|---|---|
| `log.jsonl` | one line per document: assigned GND IDs, suggestions with scores, every step, tokens, seconds, abort reason |
| `analysis.md`, `analysis.json` | written after every run: date (`created`), tool usage, F1 per decision branch, hits per tool, what `gnd_context` kept and dropped; repeat with `ie-agent analyze --name <folder>` |
| `settings.json` | model, conditions, full configuration, system prompt, prompt hash, git commit, start, every resume, end (`finished`) |
| `mapped_predictions.csv` | the pipeline format, readable by `compute_metrics.R` |
| `scores.json` | written after every run: date (`evaluated`), F1, precision, recall, R-precision and PR-AUC with casimir, as `src/compute_metrics.R` of KI-FS Prompt (R environment `r-eval`); repeat with `ie-agent score --name <folder>` |

A folder is never overwritten by accident:

| the folder already holds | what `run` does |
|---|---|
| nothing | starts a new run |
| a run with the same model, conditions and prompt | continues it |
| a run with other conditions or another prompt | stops with an error and changes nothing |
| anything, and `--force` is given | deletes `log.jsonl`, `settings.json`, `mapped_predictions.csv`, `scores.json` and the analysis of that folder and starts anew |

Every run gets its folder from its conditions, so runs with other conditions never collide. `--name X` gives a run
an own folder name instead:

| part of the name | condition | option |
|---|---|---|
| `reasoning-on/off` | reasoning of the model | `--reasoning` |
| `kag-on/off` | `gnd_context` (check in the GND graph) | `--no-kag` |
| `examples-optional/forced` | examples only on request, or always with each section | `--examples-with-sections` |
| `toc-on/off` | table of contents available | `--no-toc` |
| `_examples-gnd`, `_steps-N` | only added when not default | `--examples-gnd`, `--max-steps` |

`ie-agent --port <port> sweep --ids eval_ids.txt` runs the 8 combinations on a model server that is already running.

## When something goes wrong

| message or symptom | cause and fix |
|---|---|
| `A server already answers on port 9514` | a model server is still running. Stop it, or use it: leave out `--gpu` and give `--port 9514`. |
| `The server stopped while starting, see results/vllm.log` | the model could not start; the end of `results/vllm.log` says why. Often the GPU is busy (`nvidia-smi`). |
| `… already holds another run` | the folder holds a run with other conditions or another prompt. Use `--name` for another folder, or `--force` to delete the old results. |
| `No GND data in …` | run `ie-agent build-gnd` once. |
| `ie-agent: command not found` | not installed yet, or the terminal is older than the installation: run `uv tool install --editable .` in the `ie-agent` folder and open a new terminal. |
| `Model '…' is not under models:` | the model is unknown in `config.yaml`: add it under `models:`, or give its running server with `--port`. |
| `.venv/bin/vllm: No such file or directory` when starting vLLM by hand | vLLM lives in the environment of the main project: `~/ki-fsprompt/.venv/bin/vllm`. With `--gpu` you do not need to start it yourself. |
| many `max_steps` aborts | the model ignores the forced `finish` (seen with Muse); its documents then get all mapped candidates. |

## How it works

### Layout

```
config.yaml        all settings: models, paths, limits, defaults
prompts/           what the model reads (German, like the documents; tool names English)
src/ie_agent/
  config.py        config.yaml plus the command line options
  data.py          documents, gold, few-shot examples, GND graph, mapping (Weaviate + TEI)
  tools.py         TOOLS and KAG_TOOL as the model sees them, IE policy, tool implementations
  agent.py         the loop for one document
  runs.py          one batch run and its result folder (log, settings, predictions, scores, traces)
  server.py        starting and stopping the model server
  cli.py           the commands
  evaluation/      compute_scores.R (scores.json, casimir) and analysis.py (P/R/F1, analysis.md, analysis.json)
tests/             fake model and fake data, no GPU (uv run pytest)
eval_ids.txt       fixed evaluation set: 100 of the 827 validate documents
walkthrough.ipynb  one document step by step: data, policy, every tool call, optionally the live model
walkthrough_examples_with_sections.ipynb   the same for the variant examples_with_sections
gnd_graph/         GND knowledge for gnd_context, built with `ie-agent build-gnd` (only manifest.json is in git)
```

### IE policy (`tools.py`)

```
TITLE ──read_abstract_intro──▶ ABSTRACT ──map_to_gnd──▶ MAPPED ──read_toc──▶ TOC
```

| tool | offered when |
|---|---|
| `read_abstract_intro` | stage TITLE |
| `read_toc` | stage MAPPED (map first, then read on) |
| `get_examples` | only for sections already read (not offered with `--examples-with-sections`) |
| `map_to_gnd` | always |
| `gnd_context` | at least one candidate mapped (not offered with `--no-kag`) |
| `finish` | after `gnd_context` (without KAG: after the first mapping), or in the forced last step |

The loop (`agent.py`) adds guards: only GND IDs seen through a tool may be assigned (unknown IDs are sent back up to
twice), `finish()` is forced in the last step, and a full context window ends the run with what was mapped.

With `--examples-with-sections`, `read_abstract_intro` and `read_toc` always return the 5 examples of their
section, as in the pipeline conditions. Without KAG there is no `gnd_context`, the KAG part of the prompt is left
out and `finish` is offered as soon as something was mapped; everything else stays the same (also the 14 steps), so
the comparison isolates KAG. The parts of the system prompt that depend on the conditions are in
`prompts/system_parts.yaml`.

To change a tool, edit `TOOLS` / `KAG_TOOL` (what the model sees), `ToolRunner.run` (what happens) and, if the order
matters, `offered_tools` / `check_call` (when it is allowed). To add a model, add it under `models:` in
`config.yaml` (its Hugging Face name, a free port and the vLLM arguments, especially the tool call parser).

### GND knowledge (`gnd_graph/`)

The GND subject headings of KI-FS Prompt (`vocab/GND-Subjects-plus.ttl`, the vocabulary of the pipeline that is also
in Weaviate) provide the terms and their broader, narrower and related terms. We enriched them with official files
of the Deutsche Nationalbibliothek:

| file | source | adds |
|---|---|---|
| `authorities-gnd-sachbegriff_lds.ttl.gz` | DNB open data, https://data.dnb.de/opendata/ (CC0) | definitions, using instructions, subject category codes of subject headings |
| `authorities-gnd-geografikum_lds.ttl.gz` | DNB open data, https://data.dnb.de/opendata/ (CC0) | descriptions and subject category codes of places |
| `gnd-sc.ttl` | GND subject categories, https://d-nb.info/standards/vocab/gnd/gnd-sc.html | names of the subject categories |

The DNB dumps identify terms by GND number (e.g. `4329222-7`), KI-FS Prompt by IDN (e.g. `940053241`). They are
matched by the preferred name in Unicode NFC (the dumps store many names decomposed, e.g. "a" + combining
diaeresis), only where the name occurs once in the vocabulary and once in the dumps.

| of 309,417 terms | count |
|---|---|
| with details (subject category etc.) | 197,252 |
| with a definition | 47,127 |
| with a description (mostly places) | 14,449 |
| with using instructions | 10,951 |

Broader and narrower terms come from the vocabulary; the DNB dumps add only the kind of the relation (kind of, part
of, instance of), which `gnd_context` shows where the GND records it. If a term has more narrower terms than fit
(`max_narrower`), those named in the text come first, then those closest in meaning to the document (vectors in
Weaviate).

`gnd_context` shows definition, description and using instructions where they exist. `build-gnd` writes
`manifest.json` with URL, download date, `Last-Modified` and SHA-256 of every source, so each run can be traced back
to the exact GND version. Rebuild when the DNB publishes a new dump; results are then no longer directly comparable
with older runs.

### Differences to the original agent (`Cezars_Notitzen/agent_ie.py`)

Same logic (checked with a deterministic fake model on all 827 documents), with these intended differences:

- Tool and parameter names are English (`read_abstract_intro`, `read_toc`, `get_examples`, `map_to_gnd`,
  `gnd_context`, `finish`; sections `title`, `abstract_intro`, `toc`). Descriptions, prompts and tool results stay
  German.
- GND knowledge: two matching errors of `Cezars_Notitzen/gnd_graph.py` are fixed (names with umlauts were never
  matched; 221 of 483 subject categories lost their German name), places and using instructions were added. On the
  terms the agent queried in the Qwen IE v4 run, the share with a named subject category rose from 38 % to 92 %.
- In `gnd_context`, terms with the same frequency are ordered by GND ID. Before, the order followed Python's set
  order, which changes with every process, so a run could show different narrower terms for the same document.
- A later, fuzzier mapping onto an ID no longer replaces a better one in the stored suggestions.
- Optional conditions: without KAG, examples with every section.
