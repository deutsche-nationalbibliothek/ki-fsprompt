"""Data loading: documents and few-shot examples, the GND graph (for KAG) and mapping onto the GND via Weaviate."""
import gzip
import hashlib
import json
import pickle
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc
import requests
import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from .config import PROJECT, path

# Few-shot examples: the same ones the pipeline runs got (retrieved_examples.csv of each condition)
EXAMPLE_SOURCES = {
    "title": "title_validate",
    "abstract_intro": "abstract_intro_validate",
    "toc": "toc_strict_validate",
    "title_abstract_intro": "title_abstract_intro_validate",
    "abstract_intro_toc": "abstract_intro_toc_validate",
}
# the two combined sources are never allowed by check_call, but stay above: they are part of the prompt hash
USED_EXAMPLES = ("title", "abstract_intro", "toc")


def title_gold(cfg):
    """The ground truth of kind title: idn and uri of the assigned headings."""
    gt = pyarrow.ipc.open_file(path(cfg, "ground_truth")).read_all().to_pandas().astype(str)
    return gt[gt["kind"] == "title"]


def load_gold(cfg):
    """doc_id -> set of GND IDs assigned to it."""
    gt = title_gold(cfg)
    gold = {}
    for doc, uri in zip(gt["idn"], gt["uri"]):
        gold.setdefault(doc, set()).add(uri.split("/")[-1])
    return gold


# ---------------------------------------------------------------------------------------------------------
# Documents: title, sections, few-shot examples (loaded once for all documents)
# ---------------------------------------------------------------------------------------------------------

class Documents:
    def __init__(self, cfg):
        index = pyarrow.ipc.open_file(path(cfg, "title_index")).read_all().to_pandas()
        tsv = pd.read_csv(path(cfg, "title_tsv"), sep="\t", header=None, quoting=3, dtype=str)
        self.title = dict(zip(index["idn"].astype(str), tsv.iloc[index["location"].astype(int), 0]))
        sections = path(cfg, "sections")
        ai = pd.read_csv(sections / "abstract_intro/validate.csv", dtype=str)
        self.ids = ai["doc_id"].tolist()   # the 827 documents that exist in all conditions
        self.abstract_intro = dict(zip(ai["doc_id"], ai["text"]))
        source = pd.read_csv(sections / "abstract_intro/validate_sources.csv", dtype=str)
        self.source = dict(zip(source["doc_id"], source["source"].str.startswith("Abstract").map(
            {True: "Abstract", False: "Einleitung"})))   # shown to the model, therefore German
        toc = pd.read_csv(sections / "toc_strict/validate.csv", dtype=str).fillna({"text": ""})
        self.toc = dict(zip(toc["doc_id"], toc["text"]))
        self.examples = {}
        for section in USED_EXAMPLES:
            b = pd.read_csv(path(cfg, "results") / EXAMPLE_SOURCES[section] / "retrieved_examples.csv", dtype=str,
                            usecols=["doc_id", "prompt_text", "prompt_label_texts", "prompt_labels"])
            self.examples[section] = {doc: g[["prompt_text", "prompt_label_texts", "prompt_labels"]].values.tolist()
                                      for doc, g in b.groupby("doc_id")}


# ---------------------------------------------------------------------------------------------------------
# GND graph (from vocab/GND-Subjects-plus.ttl, SKOS) and details (definition, classification) for KAG.
# The Turtle export has a fixed layout (one block per concept, blank line in between), so a small custom parser
# is enough instead of rdflib. Built once with `ie-agent build-gnd` and stored as pickle files in gnd_graph/.
# ---------------------------------------------------------------------------------------------------------

_ID = re.compile(r"gnd:([0-9X-]+)")
_LABEL = re.compile(r'"((?:[^"\\]|\\.)*)"@de')
CATEGORY_CODE = re.compile(r"^[0-9.]+[a-z]*$")   # codes without a label in the classification vocabulary


def _parse_graph(ttl):
    pref, alt, broader, related = {}, defaultdict(int), defaultdict(set), defaultdict(set)
    for block in ttl.read_text(encoding="utf-8").split("\n\n"):
        block = block.strip()
        if not block.startswith("gnd:") or "skos:Concept" not in block.split("\n", 1)[0]:
            continue
        subject = _ID.match(block).group(1)
        # statements end with ";" at the end of a line, objects of one statement are separated by ","
        for statement in re.split(r";\s*\n", block.split("skos:Concept", 1)[1]):
            statement = statement.strip().rstrip(".").strip()
            if statement.startswith("skos:prefLabel"):
                m = _LABEL.search(statement)
                if m:
                    pref[subject] = m.group(1)
            elif statement.startswith(("skos:altLabel", "skos:hiddenLabel")):
                alt[subject] += len(_LABEL.findall(statement))
            elif statement.startswith("skos:broader"):
                broader[subject].update(_ID.findall(statement))
            elif statement.startswith("skos:related"):
                related[subject].update(_ID.findall(statement))
    narrower = defaultdict(set)
    for child, parents in broader.items():
        for p in parents:
            narrower[p].add(child)
    # make "related" symmetric (the export does not always list both directions); the direction recorded in the
    # concept's own record stays in related_out (for display: reverse links often come from persons and works)
    related_out = {k: set(v) for k, v in related.items()}
    for a, bs in list(related.items()):
        for b in bs:
            related[b].add(a)
    return {"pref": pref, "alt": dict(alt), "broader": dict(broader), "narrower": dict(narrower),
            "related": dict(related), "related_out": related_out}


_NAME = re.compile(r'gndo:preferredNameFor\w+ "((?:[^"\\]|\\.)*)"')   # subject heading, place, ...
_BROADER_KIND = re.compile(r"gndo:broaderTerm(Generic|Partitive|Instantial)\b")
_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
# dump property -> field in details (free text, several values are joined)
TEXT_FIELDS = {"gndo:definition": "definition",
               "gndo:biographicalOrHistoricalInformation": "description",
               "gndo:usingInstructions": "using_instructions"}


def nfc(text):
    """The DNB dumps store many names decomposed (NFD: "a" + combining diaeresis), KI-FS Prompt composed (NFC).
    They look the same but are different strings, so names are compared in NFC."""
    return unicodedata.normalize("NFC", text)


def _parse_details(g, dumps, classification_ttl):
    """The DNB dumps only know GND numbers (e.g. 4688372-1), our data uses IDNs (964629119): matched via the
    preferred name, only where the name occurs exactly once in our vocabulary and once in all dumps together."""
    classification = {}
    for block in classification_ttl.read_text(encoding="utf-8").split("\n\n"):
        m = re.search(r"gnd-sc#([^>]+)>\s+a skos:Concept", block)
        # the labels come in either order ("..."@en, "..."@de or the other way round): take the German one
        labels = re.search(r'skos:prefLabel\s+((?:"[^"]*"@\w+[\s,]*)+)', block)
        n = re.search(r'"([^"]*)"@de\b', labels.group(1)) if labels else None
        if m and n:
            classification[m.group(1)] = n.group(1)

    name, text, categories, current = {}, defaultdict(dict), defaultdict(list), None
    kinds, kind = defaultdict(dict), None   # GND number -> GND number of a broader term -> kind of the relation
    for dump in dumps:
        for line in gzip.open(dump, "rt", encoding="utf-8"):
            m = re.match(r"<https://d-nb.info/gnd/([0-9X-]+)>", line)
            if m:
                current = m.group(1)
            m = _BROADER_KIND.search(line)
            if m:
                kind = m.group(1)
            elif not line.startswith("    "):   # 4 spaces: more objects of the statement before
                kind = None
            if kind:
                for target in re.findall(r"gnd/([0-9X-]+)>", line[m.end():] if m else line):
                    kinds[current][target] = kind
            m = _NAME.search(line)
            if m:
                name[current] = nfc(m.group(1))
            for prop, field in TEXT_FIELDS.items():
                if prop + " " in line:
                    values = [nfc(v.replace('\\"', '"')) for v in _LITERAL.findall(line.split(prop, 1)[1])]
                    text[current][field] = " ".join(values)
            if "gndo:gndSubjectCategory" in line:
                categories[current] += re.findall(r"gnd-sc#([^>]+)>", line)

    idns_by_name = defaultdict(list)   # preferred name -> IDN
    for idn, pref in g["pref"].items():
        idns_by_name[nfc(pref)].append(idn)
    names_in_dumps = Counter(name.values())
    details = {}
    for gnd_nr, pref in name.items():
        idns = idns_by_name.get(pref, [])
        if len(idns) == 1 and names_in_dumps[pref] == 1:
            details[idns[0]] = {
                "gnd_nr": gnd_nr,
                **{field: text[gnd_nr].get(field) for field in TEXT_FIELDS.values()},
                "classification": [classification.get(k, k) for k in dict.fromkeys(categories.get(gnd_nr, []))],
            }
    idn_of = {d["gnd_nr"]: idn for idn, d in details.items()}
    for d in details.values():
        d["broader_types"] = {idn_of[t]: k for t, k in kinds.get(d["gnd_nr"], {}).items() if t in idn_of}
    return details


def training_frequency(cfg):
    """How often each GND ID was assigned in the gold (title) of the training documents (known at run time)."""
    gt = title_gold(cfg)
    train = set(pyarrow.ipc.open_file(path(cfg, "train_index")).read_all().to_pandas()["idn"].astype(str))
    gt = gt[gt["idn"].isin(train)]
    return gt["uri"].str.split("/").str[-1].value_counts().to_dict()


def gnd_files(cfg):
    """All GND files live in the project folder gnd_graph/ (downloads, parsed caches, manifest)."""
    folder = PROJECT / cfg["gnd"]["folder"]
    return {"folder": folder, "dumps": [folder / Path(url).name for url in cfg["gnd"]["dump_urls"]],
            "classification": folder / "gnd-sc.ttl", "graph": folder / "graph.pkl", "details": folder / "details.pkl",
            "manifest": folder / "manifest.json"}


def _download(url, target):
    """Download url to target and return what is needed to know later which version was used."""
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        sha256 = hashlib.sha256()
        with open(target, "wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
                sha256.update(chunk)
    return {"url": url, "file": target.name, "bytes": target.stat().st_size, "sha256": sha256.hexdigest(),
            "last_modified": response.headers.get("Last-Modified"),
            "downloaded": datetime.now().isoformat(timespec="seconds")}


def build_gnd(cfg):
    """Download the DNB files and build the graph and details from scratch. Writes manifest.json with URL, date and
    checksum of every source, so a run can be traced back to the exact GND version."""
    files = gnd_files(cfg)
    files["folder"].mkdir(parents=True, exist_ok=True)
    sources = []
    targets = list(zip(cfg["gnd"]["dump_urls"], files["dumps"])) + [(cfg["gnd"]["classification_url"],
                                                                     files["classification"])]
    for url, target in targets:
        print(f"Downloading {url} ...", flush=True)
        sources.append(_download(url, target))
    ttl = path(cfg, "gnd_ttl")
    print(f"Parsing graph from {ttl} ...", flush=True)
    g = _parse_graph(ttl)
    files["graph"].write_bytes(pickle.dumps(g))
    print("Parsing definitions, descriptions, using instructions and subject categories ...", flush=True)
    details = _parse_details(g, files["dumps"], files["classification"])
    files["details"].write_bytes(pickle.dumps(details))
    sources.append({"file": str(ttl), "bytes": ttl.stat().st_size, "note": "GND vocabulary of the pipeline (DVC)"})
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "sources": sources,
                "concepts": len(g["pref"]), "with_details": len(details),
                **{f"with_{field}": sum(1 for d in details.values() if d[field]) for field in TEXT_FIELDS.values()}}
    files["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


@dataclass
class Gnd:
    g: dict           # pref, broader, narrower, related, related_out
    details: dict     # IDN -> definition, description, using_instructions, classification
    frequency: dict   # IDN -> number of assignments in training

    @classmethod
    def load(cls, cfg):
        files = gnd_files(cfg)
        if not (files["graph"].exists() and files["details"].exists()):
            raise FileNotFoundError(f"No GND data in {files['folder']}. Build it first: uv run ie-agent build-gnd")
        g = pickle.loads(files["graph"].read_bytes())
        details = pickle.loads(files["details"].read_bytes())
        if details and "broader_types" not in next(iter(details.values())):
            raise FileNotFoundError(f"The GND data in {files['folder']} is outdated. Rebuild: ie-agent build-gnd")
        return cls(g, details, training_frequency(cfg))

    def label(self, gnd_id):
        return self.g["pref"].get(gnd_id, "?")

    def categories(self, gnd_id):
        return [c for c in self.details.get(gnd_id, {}).get("classification", []) if not CATEGORY_CODE.match(c)]

    def broader_terms(self, gnd_id, depth=3):
        """All broader terms up to the given depth (transitive)."""
        result, frontier = set(), {gnd_id}
        for _ in range(depth):
            frontier = {b for k in frontier for b in self.g["broader"].get(k, ())} - result
            result |= frontier
        return result

    def neighbours(self, gnd_id):
        """Direct neighbours in the graph: broader, narrower and related terms."""
        g = self.g
        return g["broader"].get(gnd_id, set()) | g["narrower"].get(gnd_id, set()) | g["related"].get(gnd_id, set())

    def annotation(self, gnd_id):
        """Variant A: subject category and broader terms of a heading, e.g. " (Tiermedizin; OB: Durchfall)"."""
        parts = []
        categories = self.categories(gnd_id)
        if categories:
            parts.append(categories[0])
        broader = [self.g["pref"][b] for b in sorted(self.g["broader"].get(gnd_id, ())) if b in self.g["pref"]][:2]
        if broader:
            parts.append("OB: " + ", ".join(broader))
        return f" ({'; '.join(parts)})" if parts else ""


# ---------------------------------------------------------------------------------------------------------
# Mapping onto the GND (as in src/mapping.py: hybrid search, alpha 0.7, score = hybrid score)
# ---------------------------------------------------------------------------------------------------------

class Mapper:
    def __init__(self, cfg):
        self.cfg = cfg["mapping"]
        self._local = threading.local()   # one Weaviate connection per thread

    def _weaviate(self):
        if getattr(self._local, "client", None) is None:
            self._local.client = weaviate.connect_to_local(port=self.cfg["weaviate_port"])
        return self._local.client

    def _embed(self, text):
        return np.array(requests.post(f"http://127.0.0.1:{self.cfg['tei_port']}/embed", json={"inputs": text},
                                      timeout=60).json()).reshape(-1)

    def _collection(self):
        return self._weaviate().collections.get(self.cfg["collection"])

    def closest(self, text, gnd_ids, limit):
        """Up to limit of the gnd_ids closest in meaning to text, using the vectors stored in Weaviate."""
        hits = self._collection().query.near_vector(
            near_vector=list(self._embed(unicodedata.normalize("NFC", text))), target_vector="label_vector",
            limit=limit * 5,   # one ID can have several entries in the collection
            filters=Filter.by_property("label_id").contains_any(list(gnd_ids))).objects
        return list(dict.fromkeys(hit.properties["label_id"] for hit in hits))[:limit]

    def map(self, candidate):
        query = unicodedata.normalize("NFC", candidate)
        embedding = self._embed(query)
        hit = self._collection().query.hybrid(
            query=query, vector=list(embedding), limit=1, alpha=self.cfg["alpha"],
            return_metadata=MetadataQuery(score=True), include_vector=True).objects[0]
        return {
            "candidate": candidate,
            "gnd_id": hit.properties["label_id"],
            "term": hit.properties["label_text"],
            "score": round(float(hit.metadata.score), 3),
            "cosine_similarity": float(np.dot(embedding, hit.vector["label_vector"])),
        }
