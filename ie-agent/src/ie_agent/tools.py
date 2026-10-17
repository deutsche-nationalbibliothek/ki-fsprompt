"""Tools: definitions as the model sees them, the IE policy (which tool when) and the tool implementations.

Tool and parameter names are English; descriptions and tool results stay German, like the documents and prompts.
"""
import copy
import json
import re
from enum import StrEnum

from .data import EXAMPLE_SOURCES

TOOLS = [
    {"type": "function", "function": {
        "name": "read_abstract_intro",
        "description": "Liefert das Abstract des Dokuments oder, falls es keines gibt, den Anfang der Einleitung.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_toc",
        "description": "Liefert das bereinigte Inhaltsverzeichnis des Dokuments (Kapitelüberschriften).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_examples",
        "description": "Liefert 5 ähnliche, bereits erschlossene Dokumente mit ihren GND-Schlagwörtern, gesucht über "
                       "den angegebenen Textteil dieses Dokuments.",
        "parameters": {"type": "object", "properties": {
            "section": {"type": "string", "enum": list(EXAMPLE_SOURCES)}},
            "required": ["section"]}}},
    {"type": "function", "function": {
        "name": "map_to_gnd",
        "description": "Bildet freie Schlagwort-Kandidaten auf GND-Schlagwörter ab. Liefert pro Kandidat GND-ID, "
                       "Ansetzungsform und einen Score (1,0 = exakter Treffer).",
        "parameters": {"type": "object", "properties": {
            "candidates": {"type": "array", "items": {"type": "string"}}},
            "required": ["candidates"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Gibt die finalen GND-Schlagwörter zurück. Nur GND-IDs aus vorherigen map_to_gnd-Ergebnissen.",
        "parameters": {"type": "object", "properties": {
            "gnd_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["gnd_ids"]}}},
]

# KAG (knowledge-augmented generation): tool for the GND graph
KAG_TOOL = {"type": "function", "function": {
    "name": "gnd_context",
    "description": "Liefert für GND-IDs Wissen aus der GND: Definition oder Beschreibung, Benutzungshinweis, "
                   "Sachgruppe, Ober-, Unter- und verwandte Begriffe, wie oft der Begriff in der Erschließung "
                   "vergeben wurde und wie er mit deinen anderen Kandidaten zusammenhängt. Ober- und Unterbegriffe "
                   "nennen, wo die GND es angibt, die Art der Beziehung (Art, Teil, Instanz). Die gelieferten Ober-, "
                   "Unter- und verwandten Begriffe darfst du in finish() verwenden.",
    "parameters": {"type": "object", "properties": {
        "gnd_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["gnd_ids"]}}}

ALL_TOOLS = TOOLS + [KAG_TOOL]
FINISH = "finish"


def new_state(title):
    """What the agent has done so far for one document. Policy and tools read and write here."""
    return {"read": set(), "examples": ["title"], "mapped": {}, "unknown": [], "graph": {}, "context_calls": 0,
            "text": [title], "mapped_after_abstract": False}


def allowed(state, gnd_id):
    """May this ID be assigned? Only IDs the agent saw through a tool, never from memory."""
    return gnd_id in state["mapped"] or gnd_id in state["graph"]


# ---------------------------------------------------------------------------------------------------------
# IE policy: the reading order of human indexers (from the interviews), as a state machine
#
#   TITLE ──read_abstract_intro──▶ ABSTRACT ──map_to_gnd──▶ MAPPED ──read_toc──▶ TOC
#
#   read_abstract_intro   stage TITLE
#   read_toc              stage MAPPED (map first, then read on: otherwise the model reads the TOC by reflex)
#   get_examples          only for sections already read (title examples are in the first message)
#   map_to_gnd            always
#   gnd_context           after at least one candidate was mapped
#   finish                after at least one gnd_context call, or in the forced last step
# ---------------------------------------------------------------------------------------------------------

SECTIONS = ["abstract_intro", "toc"]   # reading order after the title
READ_TOOL = {"abstract_intro": "read_abstract_intro", "toc": "read_toc"}


class Stage(StrEnum):
    TITLE = "title"
    ABSTRACT = "abstract"
    MAPPED = "mapped"
    TOC = "toc"


def stage(state):
    if "toc" in state["read"]:
        return Stage.TOC
    if state["mapped_after_abstract"]:
        return Stage.MAPPED
    if "abstract_intro" in state["read"]:
        return Stage.ABSTRACT
    return Stage.TITLE


def next_section(state):
    """The section that may be read now (or None)."""
    return {Stage.TITLE: "abstract_intro", Stage.MAPPED: "toc"}.get(stage(state))


def read_sections(state):
    return [s for s in SECTIONS if s in state["read"]]


EXAMPLES_NOTE = " Dazu kommen 5 ähnliche, bereits erschlossene Dokumente, gesucht über diesen Abschnitt."


def offered_tools(state, last_step, examples_with_sections=False, kag=True, toc=True):
    """Tool schemas the model sees in this step. With examples_with_sections the read tools return the examples of
    their section themselves, so get_examples is not offered. Without KAG there is no gnd_context, and finish is
    offered as soon as something was mapped. Without toc, read_toc does not exist."""
    names = {"map_to_gnd"}
    read_tool = READ_TOOL.get(next_section(state))
    if read_tool and (toc or read_tool != "read_toc"):
        names.add(read_tool)
    if read_sections(state) and not examples_with_sections:
        names.add("get_examples")
    if kag and state["mapped"]:
        names.add("gnd_context")
    if last_step or (state["context_calls"] if kag else state["mapped"]):
        names.add(FINISH)

    offered = copy.deepcopy([t for t in ALL_TOOLS if t["function"]["name"] in names])
    for tool in offered:
        function = tool["function"]
        if function["name"] in READ_TOOL.values() and examples_with_sections:
            function["description"] += EXAMPLES_NOTE
        if function["name"] == "get_examples":
            function["parameters"]["properties"]["section"]["enum"] = read_sections(state)
    return offered


def check_call(name, args, state, toc=True):
    """Before each call: a message for the model if the call is not allowed now, otherwise None.
    Safety net in case the model calls a tool it was not offered."""
    if name == "read_toc" and not toc:
        return f"Unbekanntes Werkzeug {name}"
    if name in READ_TOOL.values() and name != READ_TOOL.get(next_section(state)):
        return (f"{name} steht jetzt nicht zur Verfügung. Reihenfolge: Titel, dann read_abstract_intro(), dann "
                "map_to_gnd(), dann bei Bedarf read_toc(); jeder Abschnitt wird nur einmal gelesen.")
    if name == "get_examples" and args.get("section") not in read_sections(state):
        return ("Die Beispiele zum Titel hast du schon. Weitere Beispiele gibt es nur für gelesene Abschnitte: "
                f"{', '.join(read_sections(state)) or 'noch keiner gelesen'}")
    if name == "gnd_context" and not state["mapped"]:
        return "Bilde zuerst Kandidaten mit map_to_gnd() ab."
    return None


# ---------------------------------------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------------------------------------

def examples_text(examples, gnd=None, max_chars=400):
    """Few-shot examples as text; with gnd the subject headings are annotated with GND knowledge (variant A)."""
    lines = []
    for i, (text, labels, ids) in enumerate(examples, 1):
        text = re.sub(r"\s+", " ", str(text))
        if len(text) > max_chars:
            text = text[:max_chars] + " …"
        if gnd is None:
            lines.append(f"Beispiel {i}: {text}\nSchlagwörter: {labels}")
            continue
        names = [t.strip() for t in str(labels).split(";")]
        ids = [x.strip() for x in str(ids).split(",")]
        if len(ids) != len(names):   # rare: preferred names containing ";", then take names from the graph
            names = [gnd.g["pref"].get(x, x) for x in ids]
        headings = "\n".join(f"- {n}{gnd.annotation(x)}" for x, n in zip(ids, names))
        lines.append(f"Beispiel {i}: {text}\nSchlagwörter:\n{headings}")
    return "\n\n".join(lines)


# the kinds of broader relations in the DNB dumps, as shown to the model
BROADER_KIND = {"Generic": "Kandidat ist eine Art davon", "Partitive": "Kandidat ist ein Teil davon",
                "Instantial": "Kandidat ist eine Instanz davon"}
NARROWER_KIND = {"Generic": "ist eine Art des Kandidaten", "Partitive": "ist ein Teil des Kandidaten",
                 "Instantial": "ist eine Instanz des Kandidaten"}


def by_frequency(frequency, ids):
    """Most often assigned first; ties by GND ID. Without the tie-break the order would follow Python's set order,
    which changes with every process (hash randomisation), so gnd_context could show different terms per run."""
    return sorted(ids, key=lambda i: (-frequency.get(i, 0), i))


class ToolRunner:
    def __init__(self, docs, mapper, gnd, cfg):
        self.docs = docs
        self.mapper = mapper
        self.gnd = gnd
        self.cfg = cfg["agent"]
        self.min_cosine = cfg["mapping"]["min_cosine"]

    def examples(self, doc_id, section):
        gnd = self.gnd if self.cfg["examples_gnd"] else None
        return examples_text(self.docs.examples[section].get(doc_id, []), gnd, self.cfg["max_chars_example"])

    def example_ids(self, doc_id, section):
        """GND IDs of the headings in the examples of a section."""
        return [x.strip() for _, _, ids in self.docs.examples.get(section, {}).get(doc_id, [])
                for x in str(ids).split(",")]

    def step_details(self, doc_id, name, args, result, state, graph_before):
        """What a tool call yielded, for the analysis of the run."""
        if name == "map_to_gnd":
            return {"results": [{"candidate": r["kandidat"], "gnd_id": r["gnd_id"], "score": r.get("score")}
                                for r in json.loads(result)]}
        if name == "gnd_context":
            return {"shown": [i for i in state["graph"] if i not in graph_before]}
        if name == "get_examples":
            return {"example_ids": self.example_ids(doc_id, args.get("section"))}
        if name in READ_TOOL.values() and self.cfg["examples_with_sections"]:
            section = next(s for s, tool in READ_TOOL.items() if tool == name)
            return {"example_ids": self.example_ids(doc_id, section)}
        return {}

    def read(self, doc_id, section, text, state):
        """A read tool's answer; with examples_with_sections it also returns the examples of its section."""
        if not self.cfg["examples_with_sections"]:
            return text
        state["examples"].append(section)
        return f"{text}\n\nÄhnliche erschlossene Dokumente zu diesem Abschnitt:\n\n{self.examples(doc_id, section)}"

    def run(self, doc_id, name, args, state):
        """Runs an allowed tool (see check_call)."""
        if name == "read_abstract_intro":
            state["read"].add("abstract_intro")
            state["text"].append(self.docs.abstract_intro[doc_id])
            return self.read(doc_id, "abstract_intro",
                             f"[{self.docs.source[doc_id]}]\n{self.docs.abstract_intro[doc_id]}", state)
        if name == "read_toc":
            state["read"].add("toc")
            toc = self.docs.toc.get(doc_id, "")[:self.cfg["max_chars_toc"]]
            state["text"].append(toc)
            if not toc:
                return "Für dieses Dokument liegt kein Inhaltsverzeichnis vor."
            return self.read(doc_id, "toc", toc, state)
        if name == "get_examples":
            state["examples"].append(args["section"])
            return self.examples(doc_id, args["section"])
        if name == "map_to_gnd":
            return self.map(args, state)
        if name == "gnd_context" and self.cfg["kag"]:
            state["context_calls"] += 1
            ids = list(dict.fromkeys(str(i).strip() for i in args.get("gnd_ids", [])))[:self.cfg["max_context_ids"]]
            return json.dumps([self.context(i, state) for i in ids], ensure_ascii=False)
        return f"Unbekanntes Werkzeug {name}"

    def map(self, args, state):
        if "abstract_intro" in state["read"]:
            state["mapped_after_abstract"] = True   # transition ABSTRACT -> MAPPED
        results = []
        for candidate in dict.fromkeys(str(k).strip() for k in args.get("candidates", []) if str(k).strip()):
            m = self.mapper.map(candidate)
            if m["cosine_similarity"] >= self.min_cosine:
                # a later, fuzzier hit on the same ID must not replace a better one (e.g. "Galliummanganarsenid"
                # landing on "Galliumarsenid" after the exact hit)
                if m["score"] >= state["mapped"].get(m["gnd_id"], {}).get("score", -1):
                    state["mapped"][m["gnd_id"]] = m
                results.append({"kandidat": candidate, "gnd_id": m["gnd_id"], "gnd_begriff": m["term"],
                                "score": m["score"]})
            else:
                results.append({"kandidat": candidate, "gnd_id": None, "hinweis": "kein passender GND-Begriff"})
        return json.dumps(results, ensure_ascii=False)

    def context(self, gnd_id, state):
        """KAG: what the GND knows about a candidate, so the agent can check its mappings."""
        gnd, g = self.gnd, self.gnd.g
        if gnd_id not in g["pref"]:
            return {"gnd_id": gnd_id, "hinweis": "unbekannte GND-ID"}
        details = gnd.details.get(gnd_id, {})
        text = " ".join(state["text"]).lower()

        narrower = g["narrower"].get(gnd_id, set())
        in_text = by_frequency(gnd.frequency,
                               (i for i in narrower if len(gnd.label(i)) > 3 and gnd.label(i).lower() in text))
        rest = narrower - set(in_text)
        room = self.cfg["max_narrower"] - len(in_text)
        # more narrower terms than fit: the ones closest to the document, the first characters hold title and abstract
        similar = self.mapper.closest(" ".join(state["text"])[:1000], rest, room) if len(rest) > room > 0 else []
        narrower_shown = (in_text + similar + by_frequency(gnd.frequency, rest - set(similar)))[:self.cfg["max_narrower"]]
        # a broader term outside the vocabulary has no name and could never be a hit
        broader = by_frequency(gnd.frequency, (i for i in g["broader"].get(gnd_id, ()) if i in g["pref"]))[:5]
        # only relations recorded in the concept's own record (reverse direction: e.g. Adorno -> USA)
        related = by_frequency(gnd.frequency, (i for i in g["related_out"].get(gnd_id, set()) if i in gnd.details))[:5]
        for i in broader + narrower_shown + related:
            state["graph"][i] = gnd.label(i)   # may be assigned in finish() from now on

        relations = []
        own_broader = gnd.broader_terms(gnd_id, 3)
        for k in state["mapped"]:
            if k == gnd_id:
                continue
            if gnd_id in gnd.broader_terms(k, 3):
                relations.append(f"ist Oberbegriff deines Kandidaten „{gnd.label(k)}“")
            elif k in own_broader:
                relations.append(f"ist Unterbegriff deines Kandidaten „{gnd.label(k)}“")
            elif k in gnd.neighbours(gnd_id):
                relations.append(f"ist verwandt mit deinem Kandidaten „{gnd.label(k)}“")

        def relation(words, child, parent):
            """The kind of a broader relation, where the GND records it (not for the unspecific broaderTermGeneral)."""
            kind = gnd.details.get(child, {}).get("broader_types", {}).get(parent)
            return {"beziehung": words[kind]} if kind else {}

        def short(text):
            return (text[:350] + " …") if text and len(text) > 350 else text

        # description (e.g. of places) and using instructions are only shown where the GND has them
        optional = {"beschreibung": short(details.get("description")),
                    "benutzungshinweis": short(details.get("using_instructions"))}
        return {
            "gnd_id": gnd_id,
            "begriff": gnd.label(gnd_id),
            "definition": short(details.get("definition")),
            **{key: value for key, value in optional.items() if value},
            "sachgruppen": gnd.categories(gnd_id),
            "in_trainingsdokumenten_vergeben": gnd.frequency.get(gnd_id, 0),
            "oberbegriffe": [{"gnd_id": i, "begriff": gnd.label(i), **relation(BROADER_KIND, gnd_id, i)}
                             for i in broader],
            "unterbegriffe": {
                "anzahl": len(narrower),
                "auswahl": [{"gnd_id": i, "begriff": gnd.label(i), **relation(NARROWER_KIND, i, gnd_id),
                             **({"im_text": True} if i in in_text else {})}
                            for i in narrower_shown],
            },
            "verwandte_begriffe": [{"gnd_id": i, "begriff": gnd.label(i)} for i in related],
            "beziehungen_zu_deinen_kandidaten": relations,
        }
