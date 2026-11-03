"""The GND graph at run time: names, broader, narrower and related terms, details, and what the training gold says
about each heading (how often it was assigned, what it was assigned together with)."""
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pyarrow.ipc

from ..config import path
from ..data import title_gold
from .build import gnd_files

# Codes without a label in the classification vocabulary, e.g. "13.4".
CATEGORY_CODE = re.compile(r"^[0-9.]+[a-z]*$")


# ---------------------------------------------------------------------------------------------------------
# What the training gold says about each heading (known at run time, unlike the validate gold)
# ---------------------------------------------------------------------------------------------------------

def _training_gold(cfg):
    """The title gold of the training documents only."""
    gt = title_gold(cfg)
    train_index = pyarrow.ipc.open_file(path(cfg, "train_index")).read_all().to_pandas()
    train = set(train_index["idn"].astype(str))
    return gt[gt["idn"].isin(train)]


def training_frequency(cfg):
    """How often each GND ID was assigned in the gold (title) of the training documents (known at run time)."""
    gt = _training_gold(cfg)
    gnd_ids = gt["uri"].str.split("/").str[-1]
    return gnd_ids.value_counts().to_dict()


def training_cooccurrence(cfg, top=10):
    """GND ID -> the IDs most often assigned together with it in the gold (title) of the training documents."""
    gt = _training_gold(cfg)
    gnd_ids = gt["uri"].str.split("/").str[-1]

    # For every document: each of its headings counts all the others.
    counts = defaultdict(Counter)
    for _, ids in gnd_ids.groupby(gt["idn"]):
        ids = set(ids)
        for a in ids:
            counts[a].update(ids - {a})

    return {a: [b for b, _ in c.most_common(top)] for a, c in counts.items()}


# ---------------------------------------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Gnd:
    g: dict  # pref, broader, narrower, related, related_out
    details: dict  # IDN -> definition, description, using_instructions, classification
    frequency: dict  # IDN -> number of assignments in training

    @classmethod
    def load(cls, cfg):
        files = gnd_files(cfg)
        if not (files["graph"].exists() and files["details"].exists()):
            raise FileNotFoundError(f"No GND data in {files['folder']}. Build it first: uv run ie-agent build-gnd")

        g = pickle.loads(files["graph"].read_bytes())
        details = pickle.loads(files["details"].read_bytes())

        # Data built before the kinds of broader relations existed.
        if details and "broader_types" not in next(iter(details.values())):
            raise FileNotFoundError(f"The GND data in {files['folder']} is outdated. Rebuild: ie-agent build-gnd")

        return cls(g, details, training_frequency(cfg))

    def label(self, gnd_id):
        return self.g["pref"].get(gnd_id, "?")

    def categories(self, gnd_id):
        """Subject categories with a name (bare codes are left out)."""
        classification = self.details.get(gnd_id, {}).get("classification", [])
        return [c for c in classification if not CATEGORY_CODE.match(c)]

    def broader_terms(self, gnd_id, depth=3):
        """All broader terms up to the given depth (transitive)."""
        result = set()
        frontier = {gnd_id}
        for _ in range(depth):
            # One level up, without what was already found.
            frontier = {b for k in frontier for b in self.g["broader"].get(k, ())} - result
            result |= frontier
        return result

    def neighbours(self, gnd_id):
        """Direct neighbours in the graph: broader, narrower and related terms."""
        broader = self.g["broader"].get(gnd_id, set())
        narrower = self.g["narrower"].get(gnd_id, set())
        related = self.g["related"].get(gnd_id, set())
        return broader | narrower | related

    def annotation(self, gnd_id):
        """Variant A: subject category and broader terms of a heading, e.g. " (Tiermedizin; OB: Durchfall)"."""
        parts = []

        categories = self.categories(gnd_id)
        if categories:
            parts.append(categories[0])

        broader_ids = sorted(self.g["broader"].get(gnd_id, ()))
        broader = [self.g["pref"][b] for b in broader_ids if b in self.g["pref"]][:2]
        if broader:
            parts.append("OB: " + ", ".join(broader))

        return f" ({'; '.join(parts)})" if parts else ""
