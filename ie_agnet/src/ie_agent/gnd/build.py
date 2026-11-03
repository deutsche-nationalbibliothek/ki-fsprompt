"""Builds the GND knowledge for gnd_context once (ie-agent build-gnd): the graph from vocab/GND-Subjects-plus.ttl (SKOS) and
details (definition, description, using instructions, subject category, kind of broader relation) from the DNB dumps.
The Turtle export has a fixed layout (one block per concept, blank line in between), so a small custom parser is enough
instead of rdflib. Everything is stored as pickle files in gnd_graph/, with a manifest of the exact sources.
"""
import gzip
import hashlib
import json
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import requests

from ..config import PROJECT, path

# In the vocabulary (SKOS): an ID like gnd:040436802 and a German label like "Malerei"@de.
_ID = re.compile(r"gnd:([0-9X-]+)")
_LABEL = re.compile(r'"((?:[^"\\]|\\.)*)"@de')

# In the DNB dumps.
_SUBJECT = re.compile(r"<https://d-nb.info/gnd/([0-9X-]+)>")
_NAME = re.compile(r'gndo:preferredNameFor\w+ "((?:[^"\\]|\\.)*)"')  # subject heading, place, ...
_BROADER_KIND = re.compile(r"gndo:broaderTerm(Generic|Partitive|Instantial)\b")
_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')

# Dump property -> field in details (free text, several values are joined).
TEXT_FIELDS = {
    "gndo:definition": "definition",
    "gndo:biographicalOrHistoricalInformation": "description",
    "gndo:usingInstructions": "using_instructions",
}


def nfc(text):
    """The DNB dumps store many names decomposed (NFD: "a" + combining diaeresis), KI-FS Prompt composed (NFC).
    They look the same but are different strings, so names are compared in NFC."""
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------------------------------------
# The graph: names, broader, narrower and related terms (from the vocabulary of the pipeline)
# ---------------------------------------------------------------------------------------------------------

def _parse_graph(ttl):
    pref = {}
    alt = defaultdict(int)
    broader = defaultdict(set)
    related = defaultdict(set)

    for block in ttl.read_text(encoding="utf-8").split("\n\n"):
        block = block.strip()
        first_line = block.split("\n", 1)[0]
        if not block.startswith("gnd:") or "skos:Concept" not in first_line:
            continue
        subject = _ID.match(block).group(1)

        # Statements end with ";" at the end of a line, objects of one statement are separated by ",".
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

    # Narrower is broader the other way round.
    narrower = defaultdict(set)
    for child, parents in broader.items():
        for p in parents:
            narrower[p].add(child)

    # Make "related" symmetric (the export does not always list both directions). The direction recorded in the
    # concept's own record stays in related_out (for display: reverse links often come from persons and works).
    related_out = {k: set(v) for k, v in related.items()}
    for a, bs in list(related.items()):
        for b in bs:
            related[b].add(a)

    return {
        "pref": pref,
        "alt": dict(alt),
        "broader": dict(broader),
        "narrower": dict(narrower),
        "related": dict(related),
        "related_out": related_out,
    }


# ---------------------------------------------------------------------------------------------------------
# The details: definitions, using instructions, subject categories, kinds of broader relations (DNB dumps)
# ---------------------------------------------------------------------------------------------------------

def _category_names(classification_ttl):
    """Code of a subject category -> its German name."""
    names = {}
    for block in classification_ttl.read_text(encoding="utf-8").split("\n\n"):
        code = re.search(r"gnd-sc#([^>]+)>\s+a skos:Concept", block)
        # The labels come in either order ("..."@en, "..."@de or the other way round): take the German one.
        labels = re.search(r'skos:prefLabel\s+((?:"[^"]*"@\w+[\s,]*)+)', block)
        german = re.search(r'"([^"]*)"@de\b', labels.group(1)) if labels else None
        if code and german:
            names[code.group(1)] = german.group(1)
    return names


def _read_dumps(dumps):
    """Everything the dumps say per GND number. The dumps are Turtle, read line by line: a line starting with
    <https://d-nb.info/gnd/...> opens a new record, lines indented by 4 spaces continue the statement before."""
    name = {}                    # GND number -> preferred name
    text = defaultdict(dict)     # GND number -> definition, description, using_instructions
    categories = defaultdict(list)  # GND number -> codes of the subject categories
    kinds = defaultdict(dict)    # GND number -> GND number of a broader term -> kind of the relation
    current, kind = None, None

    for dump in dumps:
        for line in gzip.open(dump, "rt", encoding="utf-8"):
            subject = _SUBJECT.match(line)
            if subject:
                current = subject.group(1)

            # Kind of broader relation: holds for this line and its continuation lines.
            kind_match = _BROADER_KIND.search(line)
            if kind_match:
                kind = kind_match.group(1)
            elif not line.startswith("    "):
                kind = None
            if kind:
                targets = line[kind_match.end():] if kind_match else line
                for target in re.findall(r"gnd/([0-9X-]+)>", targets):
                    kinds[current][target] = kind

            name_match = _NAME.search(line)
            if name_match:
                name[current] = nfc(name_match.group(1))

            for prop, field in TEXT_FIELDS.items():
                if prop + " " in line:
                    values = [nfc(v.replace('\\"', '"')) for v in _LITERAL.findall(line.split(prop, 1)[1])]
                    text[current][field] = " ".join(values)

            if "gndo:gndSubjectCategory" in line:
                categories[current] += re.findall(r"gnd-sc#([^>]+)>", line)

    return name, text, categories, kinds


def _parse_details(g, dumps, classification_ttl):
    """The DNB dumps only know GND numbers (e.g. 4688372-1), our data uses IDNs (964629119): matched via the
    preferred name, only where the name occurs exactly once in our vocabulary and once in all dumps together."""
    category_names = _category_names(classification_ttl)
    name, text, categories, kinds = _read_dumps(dumps)

    # 1. Match GND numbers to IDNs via the preferred name, only where it is unique on both sides.
    idns_by_name = defaultdict(list)
    for idn, pref in g["pref"].items():
        idns_by_name[nfc(pref)].append(idn)
    names_in_dumps = Counter(name.values())

    details = {}
    for gnd_nr, pref in name.items():
        idns = idns_by_name.get(pref, [])
        if len(idns) == 1 and names_in_dumps[pref] == 1:
            codes = dict.fromkeys(categories.get(gnd_nr, []))  # without duplicates, order kept
            details[idns[0]] = {
                "gnd_nr": gnd_nr,
                **{field: text[gnd_nr].get(field) for field in TEXT_FIELDS.values()},
                "classification": [category_names.get(code, code) for code in codes],
            }

    # 2. Kinds of broader relations, translated to IDNs (only where both sides were matched).
    idn_of = {d["gnd_nr"]: idn for idn, d in details.items()}
    for d in details.values():
        d["broader_types"] = {
            idn_of[target]: kind
            for target, kind in kinds.get(d["gnd_nr"], {}).items()
            if target in idn_of
        }

    return details


# ---------------------------------------------------------------------------------------------------------
# Download and build
# ---------------------------------------------------------------------------------------------------------

def gnd_files(cfg):
    """All GND files live in the project folder gnd_graph/ (downloads, parsed caches, manifest)."""
    folder = PROJECT / cfg["gnd"]["folder"]
    return {
        "folder": folder,
        "dumps": [folder / Path(url).name for url in cfg["gnd"]["dump_urls"]],
        "classification": folder / "gnd-sc.ttl",
        "graph": folder / "graph.pkl",
        "details": folder / "details.pkl",
        "manifest": folder / "manifest.json",
    }


def _download(url, target):
    """Download url to target and return what is needed to know later which version was used."""
    sha256 = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(target, "wb") as f:
            for chunk in response.iter_content(1 << 20):  # 1 MB at a time
                f.write(chunk)
                sha256.update(chunk)

    return {
        "url": url,
        "file": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha256.hexdigest(),
        "last_modified": response.headers.get("Last-Modified"),
        "downloaded": datetime.now().isoformat(timespec="seconds"),
    }


def build_gnd(cfg):
    """Download the DNB files and build the graph and details from scratch. Writes manifest.json with URL, date and
    checksum of every source, so a run can be traced back to the exact GND version."""
    files = gnd_files(cfg)
    files["folder"].mkdir(parents=True, exist_ok=True)

    # 1. Download the dumps and the subject categories.
    targets = list(zip(cfg["gnd"]["dump_urls"], files["dumps"]))
    targets.append((cfg["gnd"]["classification_url"], files["classification"]))
    sources = []
    for url, target in targets:
        print(f"Downloading {url} ...", flush=True)
        sources.append(_download(url, target))

    # 2. The graph from the vocabulary of the pipeline.
    ttl = path(cfg, "gnd_ttl")
    print(f"Parsing graph from {ttl} ...", flush=True)
    g = _parse_graph(ttl)
    files["graph"].write_bytes(pickle.dumps(g))

    # 3. The details from the dumps.
    print("Parsing definitions, descriptions, using instructions and subject categories ...", flush=True)
    details = _parse_details(g, files["dumps"], files["classification"])
    files["details"].write_bytes(pickle.dumps(details))

    # 4. The manifest: which sources, how much was found.
    sources.append({"file": str(ttl), "bytes": ttl.stat().st_size, "note": "GND vocabulary of the pipeline (DVC)"})
    manifest = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "sources": sources,
        "concepts": len(g["pref"]),
        "with_details": len(details),
        **{f"with_{field}": sum(1 for d in details.values() if d[field]) for field in TEXT_FIELDS.values()},
    }
    files["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
