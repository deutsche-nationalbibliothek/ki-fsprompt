"""Tests with a scripted fake model and fake data: no GPU, no network."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from openai.types.chat import ChatCompletionMessageToolCall

from ie_agent.agent import FORCE_FINISH, Agent
from ie_agent.config import CONFIG
from ie_agent.data import Gnd
from ie_agent.tools import Stage, check_call, new_state, offered_tools, stage

ART, PUBLIC, PAINTING, OIL = "041143337", "040431835", "040372390", "040430000"


def call(name, **args):
    return ChatCompletionMessageToolCall(id=f"c_{name}", type="function",
                                         function={"name": name, "arguments": json.dumps(args)})


class FakeClient:
    """Plays back a fixed script of answers and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, tools, tool_choice, **kwargs):
        self.requests.append({"messages": list(messages), "tools": [t["function"]["name"] for t in tools],
                              "tool_choice": tool_choice})
        answer = self.script.pop(0)
        if isinstance(answer, Exception):
            raise answer
        text = answer if isinstance(answer, str) else ""
        calls = None if isinstance(answer, str) else answer
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=calls))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class FakeMapper:
    TABLE = {"kunst": (ART, "Kunst", 0.99), "öffentlichkeit": (PUBLIC, "Öffentlichkeit", 0.99),
             "malerei": (PAINTING, "Malerei", 0.99), "18. jahrhundert": ("981057144", "Il 18", 0.69)}

    def __init__(self):
        self.calls = []

    def map(self, candidate):
        self.calls.append(candidate)
        gnd_id, term, cos = self.TABLE.get(candidate.lower(), ("000000000", candidate, 0.5))
        return {"candidate": candidate, "gnd_id": gnd_id, "term": term, "score": 1.0, "cosine_similarity": cos}


def small_gnd():
    """Kunst ⊃ Malerei ⊃ Ölmalerei; Kunst ~ Öffentlichkeit (related)."""
    g = {"pref": {ART: "Kunst", PAINTING: "Malerei", OIL: "Ölmalerei", PUBLIC: "Öffentlichkeit"},
         "broader": {PAINTING: {ART}, OIL: {PAINTING}}, "narrower": {ART: {PAINTING}, PAINTING: {OIL}},
         "related": {ART: {PUBLIC}, PUBLIC: {ART}}, "related_out": {ART: {PUBLIC}}}
    details = {ART: {"definition": "Bildende Kunst " * 40, "classification": ["13.4", "Kunst allgemein"]},
               PUBLIC: {"definition": None, "classification": []}}
    return Gnd(g, details, {ART: 120, PAINTING: 30})


DOCS = SimpleNamespace(
    title={"1": "Malerei und Öffentlichkeit"}, abstract_intro={"1": "Es geht um Ölmalerei im Salon."},
    source={"1": "Abstract"}, toc={"1": "1 Einleitung\n2 Der Salon"},
    examples={s: {"1": [["Ein  Beispiel\nmit Umbruch", "Kunst; Malerei", f"{ART}, {PAINTING}"]]}
              for s in ["title", "abstract_intro", "toc"]})


@pytest.fixture
def build():
    def _build(script, **agent_cfg):
        cfg = yaml.safe_load(CONFIG.read_text())
        cfg["llm"]["model"] = "fake"
        cfg["agent"].update(agent_cfg)
        client, mapper = FakeClient(script), FakeMapper()
        return Agent(cfg, DOCS, mapper, small_gnd(), client=client), client, mapper
    return _build


def names(tools):
    return [t["function"]["name"] for t in tools]


# ---- IE policy ----

def test_stages_in_order(build):
    agent, _, _ = build([])
    s = new_state("t")
    assert stage(s) == Stage.TITLE
    s["read"].add("abstract_intro")
    assert stage(s) == Stage.ABSTRACT
    agent.tools.run("1", "map_to_gnd", {"candidates": []}, s)
    assert stage(s) == Stage.MAPPED
    s["read"].add("toc")
    assert stage(s) == Stage.TOC


def test_mapping_before_abstract_does_not_unlock_toc(build):
    agent, _, _ = build([])
    s = new_state("t")
    agent.tools.run("1", "map_to_gnd", {"candidates": []}, s)
    assert stage(s) == Stage.TITLE


def test_offered_tools_per_stage(build):
    agent, _, _ = build([])
    s = new_state("t")
    assert names(offered_tools(s, last_step=False)) == ["read_abstract_intro", "map_to_gnd"]
    assert "finish" in names(offered_tools(s, last_step=True))
    s["read"].add("abstract_intro")
    s["mapped"][ART] = {}
    agent.tools.run("1", "map_to_gnd", {"candidates": []}, s)
    tools = offered_tools(s, last_step=False)
    assert names(tools) == ["read_toc", "get_examples", "map_to_gnd", "gnd_context"]
    assert tools[1]["function"]["parameters"]["properties"]["section"]["enum"] == ["abstract_intro"]
    s["context_calls"] = 1
    assert "finish" in names(offered_tools(s, last_step=False))


def test_calls_that_were_not_offered_are_refused():
    s = new_state("t")
    assert "steht jetzt nicht zur Verfügung" in check_call("read_toc", {}, s)
    assert "Beispiele zum Titel hast du schon" in check_call("get_examples", {"section": "title"}, s)
    assert check_call("gnd_context", {}, s) == "Bilde zuerst Kandidaten mit map_to_gnd() ab."


# ---- agent loop ----

def test_full_run_like_an_indexer(build):
    agent, client, mapper = build([
        [call("read_abstract_intro")],
        [call("map_to_gnd", candidates=["Malerei", "Öffentlichkeit", "18. Jahrhundert", "Malerei "])],
        [call("read_toc")],
        [call("gnd_context", gnd_ids=[PAINTING])],
        [call("finish", gnd_ids=[OIL, PUBLIC])],
    ])
    r = agent.annotate("1")
    assert r["gnd_ids"] == [OIL, PUBLIC] and r["from_graph"] == [OIL]
    assert r["suggestions"][0]["source"] == "gnd_context"
    assert r["read"] == ["abstract_intro", "toc"] and r["abort"] is None
    assert mapper.calls == ["Malerei", "Öffentlichkeit", "18. Jahrhundert"]   # duplicate mapped once
    assert r["n_mapped"] == 2                                                  # 18. Jh. below min_cosine
    first = client.requests[0]["messages"][1]["content"]
    assert first.startswith("Titel: Malerei und Öffentlichkeit\n\nÄhnliche erschlossene Titel:")
    assert "Beispiel 1: Ein Beispiel mit Umbruch\nSchlagwörter: Kunst; Malerei" in first


def test_unknown_ids_are_sent_back(build):
    agent, client, _ = build([
        [call("map_to_gnd", candidates=["Kunst"])],
        [call("gnd_context", gnd_ids=[ART])],
        [call("finish", gnd_ids=[ART, "4044038-4"])],
        [call("finish", gnd_ids=[ART])],
    ])
    r = agent.annotate("1")
    assert r["gnd_ids"] == [ART] and r["rejected"] == 1 and r["unknown_ids"] == ["4044038-4"]
    assert "Gültig waren: 041143337" in client.requests[3]["messages"][-1]["content"]


def test_last_step_forces_finish(build):
    agent, client, _ = build([
        [call("map_to_gnd", candidates=["Kunst"])],
        [call("map_to_gnd", candidates=["Malerei"])],
        [call("finish", gnd_ids=[PAINTING, "made-up"])],
    ], max_steps=3)
    r = agent.annotate("1")
    assert client.requests[2]["tool_choice"] == FORCE_FINISH
    assert client.requests[2]["messages"][-1]["content"].startswith("Das ist dein letzter Schritt.")
    assert r["forced"] and r["gnd_ids"] == [PAINTING] and r["unknown_ids"] == ["made-up"]


def test_forced_finish_without_valid_ids_falls_back_to_all_mapped(build):
    agent, _, _ = build([[call("map_to_gnd", candidates=["Kunst", "Malerei"])],
                         [call("finish", gnd_ids=["made-up"])]], max_steps=2)
    r = agent.annotate("1")
    assert r["abort"] == "forced_without_valid_ids" and r["gnd_ids"] == [ART, PAINTING]


def test_answer_without_tool_is_asked_again(build):
    agent, client, _ = build([[call("map_to_gnd", candidates=["Kunst"])], "Kunst passt.",
                              [call("finish", gnd_ids=[ART])]])
    r = agent.annotate("1")
    assert client.requests[2]["tool_choice"] == FORCE_FINISH and r["gnd_ids"] == [ART]


def test_two_answers_without_tool_abort(build):
    agent, _, _ = build(["first", "second"])
    r = agent.annotate("1")
    assert r["abort"] == "no_final_answer" and r["last_answer"] == "second" and r["gnd_ids"] == []


# ---- tools ----

def test_gnd_context_shows_narrower_term_from_the_text(build):
    agent, _, _ = build([])
    s = new_state("Ölmalerei im Salon")
    s["mapped"][ART] = {}
    k = agent.tools.context(PAINTING, s)
    assert k["unterbegriffe"]["auswahl"] == [{"gnd_id": OIL, "begriff": "Ölmalerei", "im_text": True}]
    assert k["beziehungen_zu_deinen_kandidaten"] == ["ist Unterbegriff deines Kandidaten „Kunst“"]
    assert OIL in s["graph"]   # shown through the graph: may now be assigned
    art = agent.tools.context(ART, s)
    assert art["definition"].endswith(" …") and art["sachgruppen"] == ["Kunst allgemein"]
    assert agent.tools.context("999", s) == {"gnd_id": "999", "hinweis": "unbekannte GND-ID"}


def test_examples_with_gnd_annotation(build):
    agent, _, _ = build([], examples_gnd=True)
    text = agent.tools.examples("1", "title")
    assert "- Kunst (Kunst allgemein)" in text and "- Malerei (OB: Kunst)" in text


def test_ties_are_sorted_by_id():
    from ie_agent.tools import by_frequency
    assert by_frequency({"b": 2}, {"c", "a", "b"}) == ["b", "a", "c"]


# ---- GND details from the DNB dumps ----

def test_details_match_decomposed_names_and_read_new_fields(tmp_path):
    import gzip
    import unicodedata

    from ie_agent.data import _parse_details
    nfd = unicodedata.normalize("NFD", "Militärsoziologie")
    (tmp_path / "sach.ttl.gz").write_bytes(gzip.compress(f'''<https://d-nb.info/gnd/1-1> gndo:gndIdentifier "1-1";
  gndo:preferredNameForTheSubjectHeading "{nfd}";
  gndo:gndSubjectCategory <https://d-nb.info/standards/vocab/gnd/gnd-sc#9.2>;
<https://d-nb.info/gnd/1-1> gndo:usingInstructions "Für Wehrsoziologie benutze diesen Begriff"@de .
<https://d-nb.info/gnd/2-2> gndo:preferredNameForTheSubjectHeading "Bank";
<https://d-nb.info/gnd/3-3> gndo:preferredNameForTheSubjectHeading "Bank";
'''.encode()))
    (tmp_path / "geo.ttl.gz").write_bytes(gzip.compress(b'''<https://d-nb.info/gnd/4-4> gndo:gndIdentifier "4-4";
  gndo:preferredNameForThePlaceOrGeographicName "Ochtrup";
<https://d-nb.info/gnd/4-4> gndo:biographicalOrHistoricalInformation "Stadt im Kreis Steinfurt"@de .
'''))
    (tmp_path / "sc.ttl").write_text('<https://d-nb.info/standards/vocab/gnd/gnd-sc#9.2>\n    a skos:Concept ;\n'
                                     '    skos:prefLabel "Social sciences"@en, "Soziologie"@de .\n', encoding="utf-8")
    g = {"pref": {"A": "Militärsoziologie", "B": "Bank", "C": "Ochtrup"}}
    d = _parse_details(g, [tmp_path / "sach.ttl.gz", tmp_path / "geo.ttl.gz"], tmp_path / "sc.ttl")
    assert d["A"]["classification"] == ["Soziologie"]                       # NFD name matched
    assert d["A"]["using_instructions"] == "Für Wehrsoziologie benutze diesen Begriff"
    assert d["C"]["description"] == "Stadt im Kreis Steinfurt"              # from the second dump
    assert "B" not in d                                                     # ambiguous in the dumps: no match


def test_gnd_context_shows_using_instructions_only_where_present(build):
    agent, _, _ = build([])
    agent.tools.gnd.details[PUBLIC] = {"definition": None, "using_instructions": "Nur für die Sphäre",
                                       "classification": []}
    s = new_state("t")
    assert agent.tools.context(PUBLIC, s)["benutzungshinweis"] == "Nur für die Sphäre"
    assert "benutzungshinweis" not in agent.tools.context(ART, s)


def test_trace_records_reasoning_calls_and_results(build):
    agent, client, _ = build([
        [call("map_to_gnd", candidates=["Kunst"])],
        [call("gnd_context", gnd_ids=[ART])],
        [call("finish", gnd_ids=[ART])],
    ])
    original = client.create

    def with_reasoning(**kwargs):   # vLLM puts the reasoning next to the answer
        response = original(**kwargs)
        response.choices[0].message.reasoning_content = f"thinking in step {len(client.requests)}"
        return response
    client.chat.completions.create = with_reasoning
    r = agent.annotate("1", trace=True)
    steps = r["trace"]["steps"]
    assert [s["reasoning"] for s in steps] == ["thinking in step 1", "thinking in step 2", "thinking in step 3"]
    assert steps[0]["offered"] == ["read_abstract_intro", "map_to_gnd"]
    assert steps[0]["calls"] == [{"tool": "map_to_gnd", "args": '{"candidates": ["Kunst"]}'}]
    assert '"gnd_begriff": "Kunst"' in steps[0]["results"][0]["content"]


def test_examples_with_sections(build):
    agent, client, _ = build([
        [call("read_abstract_intro")],
        [call("map_to_gnd", candidates=["Kunst"])],
        [call("read_toc")],
        [call("gnd_context", gnd_ids=[ART])],
        [call("finish", gnd_ids=[ART])],
    ], examples_with_sections=True)
    r = agent.annotate("1")
    assert "automatisch 5 ähnliche" in agent.system_prompt
    assert all("get_examples" not in req["tools"] for req in client.requests)
    abstract = client.requests[1]["messages"][-1]["content"]
    assert abstract.startswith("[Abstract]\nEs geht um Ölmalerei im Salon.\n\nÄhnliche erschlossene Dokumente")
    assert "Beispiel 1: Ein Beispiel mit Umbruch" in abstract
    assert r["examples"] == ["title", "abstract_intro", "toc"]


def test_without_option_read_tools_return_only_the_section(build):
    agent, client, _ = build([[call("read_abstract_intro")], [call("map_to_gnd", candidates=["Kunst"])]], max_steps=2)
    agent.annotate("1")
    tool_results = [m["content"] for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_results == ["[Abstract]\nEs geht um Ölmalerei im Salon."]
    assert "get_examples" in client.requests[1]["tools"]


def test_fuzzy_hit_does_not_replace_exact_hit_on_same_id(build):
    agent, _, mapper = build([])
    mapper.TABLE = {**FakeMapper.TABLE, "ölbild": (ART, "Kunst", 0.95)}
    s = new_state("t")
    agent.tools.run("1", "map_to_gnd", {"candidates": ["Kunst"]}, s)
    original = mapper.map
    mapper.map = lambda c: {**original(c), "score": 0.7}
    agent.tools.run("1", "map_to_gnd", {"candidates": ["Ölbild"]}, s)
    assert s["mapped"][ART]["candidate"] == "Kunst" and s["mapped"][ART]["score"] == 1.0


# ---- conditions and result folders ----

def test_condition_names():
    from ie_agent.runs import condition_name
    cfg = yaml.safe_load(CONFIG.read_text())
    assert condition_name(cfg) == "reasoning-off_kag-on_toc-on_examples-optional"
    cfg["llm"]["reasoning"] = True
    cfg["agent"].update(kag=False, examples_with_sections=True, max_steps=10)
    assert condition_name(cfg) == "reasoning-on_kag-off_toc-on_examples-forced_steps-10"


def test_without_kag(build):
    agent, client, _ = build([
        [call("map_to_gnd", candidates=["Kunst"])],
        [call("finish", gnd_ids=[ART])],
    ], kag=False)
    r = agent.annotate("1")
    assert "gnd_context" not in agent.system_prompt and "6. Rufe zum Schluss finish()" in agent.system_prompt
    assert all("gnd_context" not in req["tools"] for req in client.requests)
    assert "finish" in client.requests[1]["tools"]          # offered as soon as something was mapped
    assert r["gnd_ids"] == [ART] and r["n_context"] == 0
    assert agent.tools.run("1", "gnd_context", {"gnd_ids": [ART]}, new_state("t")) == "Unbekanntes Werkzeug gnd_context"


def _settings(**kw):
    return {"model": "x/Fake", "conditions": "reasoning-off_kag-on_examples-optional", "prompt_hash": "h1",
            "started": "now", **kw}


def test_run_goes_into_a_folder_named_after_its_conditions():
    from ie_agent.runs import condition_name, run_folder
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["agent"]["toc"] = False
    folder = run_folder(cfg, "Qwen/Qwen3.8-27B-FP8")
    assert folder == Path("results/Qwen3.8-27B-FP8/reasoning-off_kag-on_toc-off_examples-optional")
    assert folder.name == condition_name(cfg)
    assert run_folder(cfg, "Qwen/Qwen3.8-27B-FP8", "own") == Path("results/Qwen3.8-27B-FP8/own")


def test_same_run_is_resumed(tmp_path):
    from ie_agent.runs import prepare_folder
    (tmp_path / "settings.json").write_text(json.dumps(_settings(started="yesterday", finished="old")))
    (tmp_path / "log.jsonl").write_text('{"doc_id": "1"}\n')
    kept = prepare_folder(tmp_path, _settings())
    assert kept["started"] == "yesterday" and kept["resumed"] == ["now"] and "finished" not in kept
    assert (tmp_path / "log.jsonl").read_text() == '{"doc_id": "1"}\n'


def test_other_run_is_refused_and_nothing_changes(tmp_path):
    from ie_agent.runs import prepare_folder
    (tmp_path / "settings.json").write_text(json.dumps(_settings()))
    (tmp_path / "log.jsonl").write_text('{"doc_id": "1"}\n')
    for other in (_settings(conditions="reasoning-on_kag-on_examples-optional"), _settings(prompt_hash="h2")):
        with pytest.raises(SystemExit, match="already holds another run"):
            prepare_folder(tmp_path, other)
    assert json.loads((tmp_path / "settings.json").read_text())["prompt_hash"] == "h1"
    assert (tmp_path / "log.jsonl").read_text() == '{"doc_id": "1"}\n'


def test_log_without_settings_is_refused(tmp_path):
    from ie_agent.runs import prepare_folder
    (tmp_path / "log.jsonl").write_text('{"doc_id": "1"}\n')
    with pytest.raises(SystemExit):
        prepare_folder(tmp_path, _settings())


def test_force_deletes_only_the_run_files(tmp_path):
    from ie_agent.runs import prepare_folder
    for name in ("settings.json", "log.jsonl", "mapped_predictions.csv"):
        (tmp_path / name).write_text(json.dumps(_settings()) if name == "settings.json" else "x")
    (tmp_path / "other_run").mkdir()
    (tmp_path / "other_run" / "log.jsonl").write_text("keep")
    settings = prepare_folder(tmp_path, _settings(prompt_hash="h2"), force=True)
    assert settings["prompt_hash"] == "h2"
    assert not (tmp_path / "log.jsonl").exists() and not (tmp_path / "mapped_predictions.csv").exists()
    assert (tmp_path / "other_run" / "log.jsonl").read_text() == "keep"


def test_sweep_runs_every_combination_once(monkeypatch):
    from ie_agent import cli
    seen = []
    monkeypatch.setattr(cli, "Documents", lambda cfg: SimpleNamespace(ids=["1", "2"]))
    monkeypatch.setattr(cli, "Mapper", lambda cfg: None)
    monkeypatch.setattr(cli.Gnd, "load", classmethod(lambda cls, cfg: None))
    monkeypatch.setattr(cli, "Agent", lambda cfg, *rest: Agent({**cfg, "llm": {**cfg["llm"], "model": "m"}}, *rest,
                                                                client=SimpleNamespace()))
    monkeypatch.setattr(cli, "run_batch", lambda cfg, agent, ids, name=None, force=False: seen.append(
        (cli.condition_name(cfg), agent.max_tokens, ids)))
    monkeypatch.setattr(cli, "summary", lambda args: None)
    cli.main(["sweep", "--n", "2"])
    assert len({name for name, _, _ in seen}) == 8
    assert all(tokens == (8192 if name.startswith("reasoning-on") else 2048) for name, tokens, _ in seen)
    assert all(ids == ["1", "2"] for _, _, ids in seen)


# ---- starting and stopping the vLLM server ----

FAKE_VLLM = """#!/usr/bin/env python3
import http.server, os, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
open(os.environ["FAKE_OUT"], "w").write(os.environ.get("CUDA_VISIBLE_DEVICES", "") + " " + sys.argv[2])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data": []}')
    def log_message(self, *a): pass
http.server.HTTPServer(("localhost", port), H).serve_forever()
"""


def test_server_is_started_on_the_gpu_and_always_stopped(tmp_path, monkeypatch):
    import socket

    from ie_agent.server import _answers, vllm_server
    fake = tmp_path / "vllm"
    fake.write_text(FAKE_VLLM)
    fake.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "out"))
    s = {"vllm": str(fake), "name": "Some/Model", "port": port, "args": "--x 1", "log": str(tmp_path / "v.log"),
         "timeout": 30}
    with pytest.raises(RuntimeError):
        with vllm_server(s, 3) as url:
            assert _answers(url)
            assert (tmp_path / "out").read_text() == "3 Some/Model"   # right GPU and model
            raise RuntimeError("the run fails")
    assert not _answers(f"http://localhost:{port}/v1")                 # stopped anyway


def test_default_is_all_documents():
    from ie_agent.cli import select_ids
    docs = SimpleNamespace(ids=[str(i) for i in range(10)])
    args = SimpleNamespace(ids=None, n=None)
    assert select_ids(args, {"run": {"seed": 1}}, docs) == docs.ids
    assert len(select_ids(SimpleNamespace(ids=None, n=3), {"run": {"seed": 1}}, docs)) == 3


def test_model_from_config_sets_name_and_port():
    from ie_agent.cli import parse_args
    from ie_agent.config import load_config
    cfg = load_config(parse_args(["--model", "muse", "summary"]))
    assert cfg["llm"]["model"] == "RedHatAI/Muse-Glimmer-30B-FP8-block"
    assert cfg["llm"]["base_url"] == "http://localhost:9513/v1"
    cfg = load_config(parse_args(["--port", "9600", "summary"]))
    assert cfg["llm"]["model"] == "Qwen/Qwen3.8-27B-FP8" and cfg["llm"]["base_url"] == "http://localhost:9600/v1"
    with pytest.raises(SystemExit, match="Give its server with --port"):
        load_config(parse_args(["--model", "Unknown/Model", "summary"]))


# ---- scores.json ----

FAKE_RSCRIPT = """#!/usr/bin/env python3
import json, sys
open(sys.argv[sys.argv.index("--output") + 1], "w").write(json.dumps({"f1@100": {"value": 0.5}}))
"""


def _scored_run(tmp_path, rscript):
    from ie_agent import runs
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "log.jsonl").write_text(json.dumps({"doc_id": "1", "suggestions": []}) + "\n")
    (folder / "scores.json").write_text("old")
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["scoring"]["rscript"] = str(rscript)
    cfg["run"]["output"] = str(tmp_path)
    return runs.write_scores(cfg, folder), folder


def test_scores_are_written_after_a_run(tmp_path):
    fake = tmp_path / "Rscript"
    fake.write_text(FAKE_RSCRIPT)
    fake.chmod(0o755)
    scores, folder = _scored_run(tmp_path, fake)
    assert scores["f1@100"]["value"] == 0.5 and json.loads((folder / "scores.json").read_text()) == scores
    assert scores["evaluated"].startswith("20")


def test_failed_scoring_keeps_the_run_and_no_old_scores(tmp_path, capsys):
    fake = tmp_path / "Rscript"
    fake.write_text("#!/bin/sh\necho kaputt >&2\nexit 1\n")
    fake.chmod(0o755)
    scores, folder = _scored_run(tmp_path, fake)
    assert scores is None and "Scoring failed" in capsys.readouterr().out
    assert (folder / "log.jsonl").exists() and not (folder / "scores.json").exists()


# ---- without table of contents ----

def test_without_toc(build):
    agent, client, _ = build([
        [call("read_abstract_intro")],
        [call("map_to_gnd", candidates=["Kunst"])],
        [call("read_toc")],
        [call("gnd_context", gnd_ids=[ART])],
        [call("finish", gnd_ids=[ART])],
    ], toc=False)
    r = agent.annotate("1")
    assert "read_toc" not in agent.system_prompt and "Inhaltsverzeichnis" not in agent.system_prompt
    assert all("read_toc" not in req["tools"] for req in client.requests)
    assert r["read"] == ["abstract_intro"] and r["steps"][2]["refused"]
    assert r["gnd_ids"] == [ART]


def test_default_prompt_hash_is_unchanged(build):
    agent, _, _ = build([])
    assert agent.prompt_hash == "f8f61c84d24a"   # hash of the full Qwen run of 30.10.2026


# ---- analysis of a run ----

def test_analysis_counts_tools_branches_and_hits(build):
    from ie_agent.evaluation.analysis import analyze, to_markdown
    agent, _, _ = build([
        [call("read_abstract_intro")],
        [call("map_to_gnd", candidates=["Kunst", "Malerei", "18. Jahrhundert"])],
        [call("read_toc")],
        [call("get_examples", section="toc")],
        [call("map_to_gnd", candidates=["Öffentlichkeit"])],
        [call("gnd_context", gnd_ids=[PAINTING])],
        [call("finish", gnd_ids=[OIL, ART])],
    ])
    r = agent.annotate("1")
    gold = {"1": {OIL, PAINTING, "999"}}
    a = analyze([r, {"doc_id": "2", "error": "x"}], gold)
    assert a["docs"] == 1 and a["errors"] == 1
    assert a["tools"]["map_to_gnd"] == {"calls": 2, "docs": 1, "per_doc": 2.0, "refused": 0}
    assert a["branches"]["reading"]["title + abstract + toc"]["docs"] == 1
    hits = a["tool_hits"]
    assert hits["map_to_gnd"] == {"candidates": 4, "accepted": 3, "rejected_low_similarity": 1}
    assert hits["first_mapped"]["after abstract"]["mapped"] == 2     # Kunst, Malerei
    assert hits["first_mapped"]["after toc"]["mapped"] == 1          # Öffentlichkeit
    k = hits["gnd_context"]
    assert (k["kept_wrong"], k["dropped_correct"], k["dropped_wrong"]) == (1, 1, 1)
    assert k["checked_dropped_correct"] == 1                         # Malerei was checked, then replaced
    assert (k["assigned_from_graph"], k["assigned_from_graph_correct"]) == (1, 1)
    assert hits["examples"]["toc"]["docs"] == 1 and hits["examples"]["toc"]["docs_with_gold"] == 1
    assert "## Tool usage" in to_markdown(a, "test")
    assert "Created: 2026-10-30T10:00:00" in to_markdown({"created": "2026-10-30T10:00:00", **a}, "test")


# ---- GND graph: what gnd_context shows ----

def test_narrower_terms_that_do_not_fit_are_chosen_by_meaning(build):
    agent, _, mapper = build([], max_narrower=2)
    g = agent.tools.gnd.g
    g["pref"].update(a="Aaa", b="Bbb", c="Ccc")
    g["narrower"][ART] = {"a", "b", "c", OIL}
    seen = []
    mapper.closest = lambda text, ids, limit: seen.append((text, set(ids), limit)) or ["c"]
    shown = agent.tools.context(ART, new_state("Ölmalerei"))["unterbegriffe"]["auswahl"]
    assert [i["gnd_id"] for i in shown] == [OIL, "c"] and shown[0]["im_text"]       # in the text first, then by meaning
    assert seen == [("Ölmalerei", {"a", "b", "c"}, 1)]


def test_no_similarity_search_when_all_narrower_terms_fit(build):
    agent, _, mapper = build([])
    mapper.closest = lambda *a: pytest.fail("not needed")
    assert len(agent.tools.context(ART, new_state("t"))["unterbegriffe"]["auswahl"]) == 1


def test_broader_term_outside_the_vocabulary_is_not_offered(build):
    agent, _, _ = build([])
    agent.tools.gnd.g["broader"][PAINTING] = {ART, "outside"}
    s = new_state("t")
    assert [i["gnd_id"] for i in agent.tools.context(PAINTING, s)["oberbegriffe"]] == [ART]
    assert "outside" not in s["graph"]


def test_kind_of_relation_is_shown_in_both_directions(build):
    agent, _, _ = build([])
    agent.tools.gnd.details[PAINTING] = {"broader_types": {ART: "Generic"}, "classification": []}
    s = new_state("t")
    assert agent.tools.context(PAINTING, s)["oberbegriffe"][0]["beziehung"] == "Kandidat ist eine Art davon"
    assert agent.tools.context(ART, s)["unterbegriffe"]["auswahl"][0]["beziehung"] == "ist eine Art des Kandidaten"


def test_kinds_of_broader_terms_are_read_from_the_dumps(tmp_path):
    import gzip

    from ie_agent.data import _parse_details
    (tmp_path / "d.ttl.gz").write_bytes(gzip.compress(b'''<https://d-nb.info/gnd/1-1> gndo:gndIdentifier "1-1";
  gndo:preferredNameForTheSubjectHeading "Kind";
  gndo:broaderTermPartitive <https://d-nb.info/gnd/2-2>,
    <https://d-nb.info/gnd/3-3>;
  gndo:broaderTermGeneral <https://d-nb.info/gnd/4-4>;
  gndo:broaderTermGeneric <https://d-nb.info/gnd/5-5>;
  wdrs:describedby <https://d-nb.info/gnd/6-6/about> .

<https://d-nb.info/gnd/2-2> gndo:preferredNameForTheSubjectHeading "Teil";
<https://d-nb.info/gnd/3-3> gndo:preferredNameForTheSubjectHeading "Teil2";
<https://d-nb.info/gnd/4-4> gndo:preferredNameForTheSubjectHeading "Allgemein";
<https://d-nb.info/gnd/6-6> gndo:preferredNameForTheSubjectHeading "Sechs";
'''))
    (tmp_path / "sc.ttl").write_text("", encoding="utf-8")
    g = {"pref": {"K": "Kind", "T": "Teil", "U": "Teil2", "A": "Allgemein", "S": "Sechs"}}
    d = _parse_details(g, [tmp_path / "d.ttl.gz"], tmp_path / "sc.ttl")
    # General has no kind, 5-5 is not in the vocabulary, 6-6 is not a broader term
    assert d["K"]["broader_types"] == {"T": "Partitive", "U": "Partitive"}


def test_old_gnd_data_must_be_rebuilt(tmp_path, monkeypatch):
    import pickle

    from ie_agent import data
    files = {"folder": tmp_path, "graph": tmp_path / "g", "details": tmp_path / "d"}
    files["graph"].write_bytes(pickle.dumps({}))
    files["details"].write_bytes(pickle.dumps({"1": {"classification": []}}))
    monkeypatch.setattr(data, "gnd_files", lambda cfg: files)
    with pytest.raises(FileNotFoundError, match="build-gnd"):
        Gnd.load({})


# ---- fixes after the review ----

def _finish_raw(raw):
    return ChatCompletionMessageToolCall(id="c", type="function", function={"name": "finish", "arguments": raw})


@pytest.mark.parametrize("last", [[_finish_raw('{"gnd')], [call("finish", gnd_ids=["x"])]])
def test_finish_without_valid_ids_falls_back_to_all_mapped(build, last):
    agent, _, _ = build([[call("map_to_gnd", candidates=["Kunst"])], [call("gnd_context", gnd_ids=[ART])],
                         *[last] * 3], max_rejections=2)
    r = agent.annotate("1")
    assert r["gnd_ids"] == [ART] and r["abort"] == "finish_without_valid_ids"


def test_text_before_any_mapping_does_not_end_the_document(build):
    agent, client, _ = build(["Ich denke nach.", [call("map_to_gnd", candidates=["Kunst"])],
                              [call("gnd_context", gnd_ids=[ART])], [call("finish", gnd_ids=[ART])]])
    r = agent.annotate("1")
    assert client.requests[1]["tool_choice"] == "auto" and client.requests[2]["tool_choice"] == "auto"
    assert "map_to_gnd" in client.requests[1]["messages"][-1]["content"]
    assert r["gnd_ids"] == [ART] and r["abort"] is None and not r["forced"]


def test_other_token_budget_or_mapping_is_refused(tmp_path):
    from ie_agent.runs import prepare_folder
    config = {"llm": {"max_tokens": 2048}, "mapping": {"alpha": 0.7, "min_cosine": 0.7, "weaviate_port": 1}}
    (tmp_path / "settings.json").write_text(json.dumps(_settings(config=config)))
    other_port = {**config, "mapping": {**config["mapping"], "weaviate_port": 2}}
    assert prepare_folder(tmp_path, _settings(config=other_port))["resumed"] == ["now"]
    for other in ({**config, "llm": {"max_tokens": 512}}, {**config, "mapping": {**config["mapping"], "alpha": 0.5}}):
        with pytest.raises(SystemExit, match="already holds another run"):
            prepare_folder(tmp_path, _settings(config=other))


def _old_run(tmp_path):
    folder = tmp_path / "m" / "old"
    folder.mkdir(parents=True)
    (folder / "protokoll.jsonl").write_text(json.dumps({"doc_id": "1", "gnd_ids": [ART], "n_schritte": 3}) + "\n")
    return folder


def test_new_run_ignores_an_old_log(tmp_path, monkeypatch):
    from ie_agent import runs
    monkeypatch.setattr(runs, "write_scores", lambda *a: None)
    monkeypatch.setattr(runs, "write_analysis", lambda *a: None)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["run"]["output"] = str(tmp_path)
    _old_run(tmp_path)
    done = []
    agent = SimpleNamespace(model="x/m", prompt_hash="h", system_prompt="s",
                            annotate=lambda d: done.append(d) or {"doc_id": d, "suggestions": []})
    runs.run_batch(cfg, agent, ["1", "2"], name="old")
    assert sorted(done) == ["1", "2"]


def test_eval_reads_an_old_log(tmp_path, monkeypatch, capsys):
    from ie_agent import cli
    _old_run(tmp_path)
    monkeypatch.setattr(cli, "load_gold", lambda cfg: {"1": {ART}})
    cli.main(["--output", str(tmp_path), "eval", "--name", "m/old"])
    assert "1.000" in capsys.readouterr().out


def test_summary_skips_a_run_without_documents(tmp_path, monkeypatch, capsys):
    from ie_agent import cli
    folder = tmp_path / "m" / "empty"
    folder.mkdir(parents=True)
    (folder / "log.jsonl").write_text("")
    monkeypatch.setattr(cli, "load_gold", lambda cfg: {})
    cli.main(["--output", str(tmp_path), "summary"])
    assert "No runs" in capsys.readouterr().out


def test_analysis_of_an_old_log_has_no_hits_per_tool():
    from ie_agent.evaluation.analysis import analyze, to_markdown
    e = {"doc_id": "1", "gnd_ids": ["a"], "steps": [{"tool": "finish", "args": {}}], "n_steps": 1, "tokens_in": 1,
         "read": [], "forced": False, "abort": None, "from_graph": []}
    a = analyze([e], {"1": {"a"}})
    assert a["tool_hits"] is None and "Not available" in to_markdown(a, "old")
