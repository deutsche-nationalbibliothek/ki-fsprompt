"""Tools: definitions as the model sees them, the IE policy (which tool when) and the tool implementations.

Tool and parameter names are English; descriptions and tool results stay German, like the documents and prompts.
"""
import copy
import json
import re
from enum import StrEnum

from .data import EXAMPLE_SOURCES
from .gnd import gnd_context

# ---------------------------------------------------------------------------------------------------------
# Tool definitions: exactly what the model sees (part of the prompt hash, so any change starts new runs)
# ---------------------------------------------------------------------------------------------------------


def _tool(name, description, parameters=None, required=()):
    """A tool schema in the OpenAI format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters or {},
                "required": list(required),
            },
        },
    }


STRING_LIST = {"type": "array", "items": {"type": "string"}}

TOOLS = [
    _tool(
        "read_abstract_intro",
        "Liefert das Abstract des Dokuments oder, falls es keines gibt, den Anfang der Einleitung.",
    ),
    _tool(
        "read_toc",
        "Liefert das bereinigte Inhaltsverzeichnis des Dokuments (Kapitelüberschriften).",
    ),
    _tool(
        "get_examples",
        "Liefert 5 ähnliche, bereits erschlossene Dokumente mit ihren GND-Schlagwörtern, gesucht über "
        "den angegebenen Textteil dieses Dokuments.",
        {"section": {"type": "string", "enum": list(EXAMPLE_SOURCES)}},
        required=["section"],
    ),
    _tool(
        "map_to_gnd",
        "Bildet freie Schlagwort-Kandidaten auf GND-Schlagwörter ab. Liefert pro Kandidat GND-ID, "
        "Ansetzungsform und einen Score (1,0 = exakter Treffer).",
        {"candidates": STRING_LIST},
        required=["candidates"],
    ),
    _tool(
        "finish",
        "Gibt die finalen GND-Schlagwörter zurück. Nur GND-IDs aus vorherigen map_to_gnd-Ergebnissen.",
        {"gnd_ids": STRING_LIST},
        required=["gnd_ids"],
    ),
]

# gnd_context: tool for the GND graph
GND_CONTEXT_TOOL = _tool(
    "gnd_context",
    "Liefert für GND-IDs Wissen aus der GND: Definition oder Beschreibung, Benutzungshinweis, "
    "Sachgruppe, Ober-, Unter- und verwandte Begriffe, wie oft der Begriff in der Erschließung "
    "vergeben wurde und wie er mit deinen anderen Kandidaten zusammenhängt. Ober- und Unterbegriffe "
    "nennen, wo die GND es angibt, die Art der Beziehung (Art, Teil, Instanz). Die gelieferten Ober-, "
    "Unter- und verwandten Begriffe darfst du in finish() verwenden.",
    {"gnd_ids": STRING_LIST},
    required=["gnd_ids"],
)

ALL_TOOLS = TOOLS + [GND_CONTEXT_TOOL]
FINISH = "finish"


# ---------------------------------------------------------------------------------------------------------
# The state of one document
# ---------------------------------------------------------------------------------------------------------

def new_state(title):
    """What the agent has done so far for one document. Policy and tools read and write here."""
    return {
        "read": set(),  # sections read: abstract_intro, toc
        "examples": ["title"],  # sections whose examples the model got
        "mapped": {},  # GND ID -> mapping result (map_to_gnd)
        "unknown": [],  # IDs in finish() that came from no tool
        "graph": {},  # GND ID -> name, shown by gnd_context
        "context_calls": 0,
        "text": [title],  # everything read so far (for gnd_context: narrower terms in the text)
        "mapped_after_abstract": False,
    }


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

SECTIONS = ["abstract_intro", "toc"]  # reading order after the title
READ_TOOL = {"abstract_intro": "read_abstract_intro", "toc": "read_toc"}


class Stage(StrEnum):
    TITLE = "title"
    ABSTRACT = "abstract"
    MAPPED = "mapped"
    TOC = "toc"


def stage(state):
    """The stage follows from the state (checked from the last stage backwards)."""
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
    """The sections read so far, in reading order."""
    return [s for s in SECTIONS if s in state["read"]]


EXAMPLES_NOTE = " Dazu kommen 5 ähnliche, bereits erschlossene Dokumente, gesucht über diesen Abschnitt."


def offered_tools(state, last_step, examples_with_sections=False, gnd_context=True, toc=True):
    """Tool schemas the model sees in this step. With examples_with_sections the read tools return the examples of
    their section themselves, so get_examples is not offered. Without gnd_context there is no gnd_context, and finish is
    offered as soon as something was mapped. Without toc, read_toc does not exist."""
    # What has happened so far.
    next_read_tool = READ_TOOL.get(next_section(state))  # read_abstract_intro, read_toc or None
    something_read = bool(read_sections(state))
    something_mapped = bool(state["mapped"])
    checked_with_gnd_context = state["context_calls"] > 0

    # 1. For every tool: may the model use it now?
    may_use = {
        # Always.
        "map_to_gnd": True,

        # Reading: only the next section in the reading order (title -> abstract -> map -> toc).
        "read_abstract_intro": next_read_tool == "read_abstract_intro",
        "read_toc": next_read_tool == "read_toc" and toc,

        # Examples for sections already read (the title examples are in the first message).
        # Not needed when the read tools already return the examples themselves.
        "get_examples": something_read and not examples_with_sections,

        # Checking in the GND graph needs at least one mapped candidate.
        "gnd_context": gnd_context and something_mapped,

        # Finishing: after the check with gnd_context (without gnd_context: after mapping), or when forced.
        FINISH: last_step or (checked_with_gnd_context if gnd_context else something_mapped),
    }

    # 2. The schemas of these tools. A copy, because some are adapted below and ALL_TOOLS must stay unchanged.
    offered = copy.deepcopy([tool for tool in ALL_TOOLS if may_use[tool["function"]["name"]]])

    # 3. Adapt them to the state.
    for tool in offered:
        function = tool["function"]

        # The read tools say that they also return the examples of their section.
        if function["name"] in READ_TOOL.values() and examples_with_sections:
            function["description"] += EXAMPLES_NOTE

        # get_examples only accepts sections already read.
        if function["name"] == "get_examples":
            function["parameters"]["properties"]["section"]["enum"] = read_sections(state)

    return offered


def check_call(name, args, state, toc=True):
    """Before each call: a message for the model if the call is not allowed now, otherwise None.
    Safety net in case the model calls a tool it was not offered."""
    if name == "read_toc" and not toc:
        return f"Unbekanntes Werkzeug {name}"

    if name in READ_TOOL.values() and name != READ_TOOL.get(next_section(state)):
        return (
            f"{name} steht jetzt nicht zur Verfügung. Reihenfolge: Titel, dann read_abstract_intro(), dann "
            "map_to_gnd(), dann bei Bedarf read_toc(); jeder Abschnitt wird nur einmal gelesen."
        )

    if name == "get_examples" and args.get("section") not in read_sections(state):
        sections = ", ".join(read_sections(state)) or "noch keiner gelesen"
        return (
            "Die Beispiele zum Titel hast du schon. Weitere Beispiele gibt es nur für gelesene Abschnitte: "
            f"{sections}"
        )

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
        # Text on one line, cut after max_chars.
        text = re.sub(r"\s+", " ", str(text))
        if len(text) > max_chars:
            text = text[:max_chars] + " …"

        if gnd is None:
            lines.append(f"Beispiel {i}: {text}\nSchlagwörter: {labels}")
            continue

        # Variant A: one heading per line, with subject category and broader terms.
        names = [t.strip() for t in str(labels).split(";")]
        ids = [x.strip() for x in str(ids).split(",")]
        if len(ids) != len(names):
            # Rare: preferred names containing ";", then take the names from the graph.
            names = [gnd.g["pref"].get(x, x) for x in ids]
        headings = "\n".join(f"- {n}{gnd.annotation(x)}" for x, n in zip(ids, names))
        lines.append(f"Beispiel {i}: {text}\nSchlagwörter:\n{headings}")

    return "\n\n".join(lines)


class ToolRunner:
    def __init__(self, docs, mapper, gnd, cfg):
        self.docs = docs
        self.mapper = mapper
        self.gnd = gnd
        self.cfg = cfg["agent"]
        self.min_cosine = cfg["mapping"]["min_cosine"]

    # --- examples ---

    def examples(self, doc_id, section):
        gnd = self.gnd if self.cfg["examples_gnd"] else None
        return examples_text(self.docs.examples[section].get(doc_id, []), gnd, self.cfg["max_chars_example"])

    def example_ids(self, doc_id, section):
        """GND IDs of the headings in the examples of a section."""
        examples = self.docs.examples.get(section, {}).get(doc_id, [])
        return [x.strip() for _, _, ids in examples for x in str(ids).split(",")]

    # --- running a tool ---

    def run(self, doc_id, name, args, state):
        """Runs an allowed tool (see check_call)."""
        if name == "read_abstract_intro":
            return self._read_abstract_intro(doc_id, state)
        if name == "read_toc":
            return self._read_toc(doc_id, state)
        if name == "get_examples":
            state["examples"].append(args["section"])
            return self.examples(doc_id, args["section"])
        if name == "map_to_gnd":
            return self.map(args, state)
        if name == "gnd_context" and self.cfg["gnd_context"]:
            return self._gnd_context(args, state)
        return f"Unbekanntes Werkzeug {name}"

    def _read_abstract_intro(self, doc_id, state):
        state["read"].add("abstract_intro")
        text = self.docs.abstract_intro[doc_id]
        state["text"].append(text)
        # e.g. "[Einleitung]\n..."
        return self.read(doc_id, "abstract_intro", f"[{self.docs.source[doc_id]}]\n{text}", state)

    def _read_toc(self, doc_id, state):
        state["read"].add("toc")
        toc = self.docs.toc.get(doc_id, "")[:self.cfg["max_chars_toc"]]
        state["text"].append(toc)
        if not toc:
            return "Für dieses Dokument liegt kein Inhaltsverzeichnis vor."
        return self.read(doc_id, "toc", toc, state)

    def read(self, doc_id, section, text, state):
        """A read tool's answer; with examples_with_sections it also returns the examples of its section."""
        if not self.cfg["examples_with_sections"]:
            return text
        state["examples"].append(section)
        return f"{text}\n\nÄhnliche erschlossene Dokumente zu diesem Abschnitt:\n\n{self.examples(doc_id, section)}"

    def map(self, args, state):
        """map_to_gnd: each candidate onto its closest GND heading; accepted if the cosine similarity is high enough."""
        if "abstract_intro" in state["read"]:
            state["mapped_after_abstract"] = True  # transition ABSTRACT -> MAPPED

        # Without empty and repeated candidates.
        candidates = dict.fromkeys(str(k).strip() for k in args.get("candidates", []) if str(k).strip())

        results = []
        for candidate in candidates:
            m = self.mapper.map(candidate)

            if m["cosine_similarity"] < self.min_cosine:
                results.append({"kandidat": candidate, "gnd_id": None, "hinweis": "kein passender GND-Begriff"})
                continue

            # A later, fuzzier hit on the same ID must not replace a better one (e.g. "Galliummanganarsenid"
            # landing on "Galliumarsenid" after the exact hit).
            if m["score"] >= state["mapped"].get(m["gnd_id"], {}).get("score", -1):
                state["mapped"][m["gnd_id"]] = m
            results.append({
                "kandidat": candidate,
                "gnd_id": m["gnd_id"],
                "gnd_begriff": m["term"],
                "score": m["score"],
            })

        return json.dumps(results, ensure_ascii=False)

    def _gnd_context(self, args, state):
        state["context_calls"] += 1
        # Without repeated IDs, at most max_context_ids per call.
        ids = list(dict.fromkeys(str(i).strip() for i in args.get("gnd_ids", [])))[:self.cfg["max_context_ids"]]
        return json.dumps([self.context(i, state) for i in ids], ensure_ascii=False)

    def context(self, gnd_id, state):
        return gnd_context(self.gnd, gnd_id, state, self.mapper, self.cfg["max_narrower"])

    # --- for the analysis of a run (the model does not see this) ---

    def step_details(self, doc_id, name, args, result, state, graph_before):
        """What a tool call yielded, for the analysis of the run."""
        if name == "map_to_gnd":
            results = [
                {"candidate": r["kandidat"], "gnd_id": r["gnd_id"], "score": r.get("score")}
                for r in json.loads(result)
            ]
            return {"results": results}
        if name == "gnd_context":
            return {"shown": [i for i in state["graph"] if i not in graph_before]}
        if name == "get_examples":
            return {"example_ids": self.example_ids(doc_id, args.get("section"))}
        if name in READ_TOOL.values() and self.cfg["examples_with_sections"]:
            section = next(s for s, tool in READ_TOOL.items() if tool == name)
            return {"example_ids": self.example_ids(doc_id, section)}
        return {}
