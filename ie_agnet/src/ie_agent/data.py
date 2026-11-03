"""Data loading: documents, gold and few-shot examples, and mapping onto the GND via Weaviate (the GND graph is in gnd/)."""
import threading
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.ipc
import requests
import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from .config import path

# Few-shot examples: the same ones the pipeline runs got (retrieved_examples.csv of each condition).
EXAMPLE_SOURCES = {
    "title": "title_validate",
    "abstract_intro": "abstract_intro_validate",
    "toc": "toc_strict_validate",
    "title_abstract_intro": "title_abstract_intro_validate",
    "abstract_intro_toc": "abstract_intro_toc_validate",
}

# The two combined sources are never allowed by check_call, but stay above: they are part of the prompt hash.
USED_EXAMPLES = ("title", "abstract_intro", "toc")


# ---------------------------------------------------------------------------------------------------------
# Gold: the headings the indexers assigned
# ---------------------------------------------------------------------------------------------------------

def title_gold(cfg):
    """The ground truth of kind title: idn and uri of the assigned headings."""
    gt = pyarrow.ipc.open_file(path(cfg, "ground_truth")).read_all().to_pandas().astype(str)
    return gt[gt["kind"] == "title"]


def load_gold(cfg):
    """doc_id -> set of GND IDs assigned to it."""
    gt = title_gold(cfg)
    gold = {}
    for doc, uri in zip(gt["idn"], gt["uri"]):
        # https://d-nb.info/gnd/040436802 -> 040436802
        gold.setdefault(doc, set()).add(uri.split("/")[-1])
    return gold


# ---------------------------------------------------------------------------------------------------------
# Documents: title, sections, few-shot examples (loaded once for all documents)
# ---------------------------------------------------------------------------------------------------------

class Documents:
    def __init__(self, cfg):
        sections = path(cfg, "sections")
        abstract_source = cfg["agent"].get("abstract_source", "abstract_intro")

        # 1. Titles: the index says in which line of the TSV the title of a document is.
        index = pyarrow.ipc.open_file(path(cfg, "title_index")).read_all().to_pandas()
        tsv = pd.read_csv(path(cfg, "title_tsv"), sep="\t", header=None, quoting=3, dtype=str)
        titles = tsv.iloc[index["location"].astype(int), 0]
        self.title = dict(zip(index["idn"].astype(str), titles))

        # 2. Abstract or introduction (what read_abstract_intro returns).
        ai = pd.read_csv(sections / "abstract_intro/validate.csv", dtype=str)
        self.ids = ai["doc_id"].tolist()  # the 827 documents that exist in all conditions
        if abstract_source == "title_abstract_intro":
            # Condition C: title + abstract/introduction.
            ai = pd.read_csv(path(cfg, "bedingungen") / "title_abstract_intro/validate.csv", dtype=str)
        self.abstract_intro = dict(zip(ai["doc_id"], ai["text"]))

        # Was it an abstract or the introduction? Shown to the model, therefore German.
        source = pd.read_csv(sections / "abstract_intro/validate_sources.csv", dtype=str)
        is_abstract = source["source"].str.startswith("Abstract")
        self.source = dict(zip(source["doc_id"], is_abstract.map({True: "Abstract", False: "Einleitung"})))

        # 3. Table of contents (empty text: the document has none).
        toc = pd.read_csv(sections / "toc_strict/validate.csv", dtype=str).fillna({"text": ""})
        self.toc = dict(zip(toc["doc_id"], toc["text"]))

        # 4. Few-shot examples per section: doc_id -> [[text, heading names, heading IDs], ...].
        self.examples = {}
        for section in USED_EXAMPLES:
            # The examples of read_abstract_intro come from the same condition as its text.
            source = abstract_source if section == "abstract_intro" else section
            retrieved = pd.read_csv(
                path(cfg, "results") / EXAMPLE_SOURCES[source] / "retrieved_examples.csv",
                dtype=str,
                usecols=["doc_id", "prompt_text", "prompt_label_texts", "prompt_labels"],
            )
            self.examples[section] = {
                doc: group[["prompt_text", "prompt_label_texts", "prompt_labels"]].values.tolist()
                for doc, group in retrieved.groupby("doc_id")
            }


# ---------------------------------------------------------------------------------------------------------
# Mapping onto the GND (as in src/mapping.py: hybrid search, alpha 0.7, score = hybrid score)
# ---------------------------------------------------------------------------------------------------------

class Mapper:
    def __init__(self, cfg):
        self.cfg = cfg["mapping"]
        self._local = threading.local()  # one Weaviate connection per thread

    def _weaviate(self):
        if getattr(self._local, "client", None) is None:
            self._local.client = weaviate.connect_to_local(port=self.cfg["weaviate_port"])
        return self._local.client

    def _embed(self, text):
        """The embedding of a text, from Text Embeddings Inference (TEI)."""
        response = requests.post(
            f"http://127.0.0.1:{self.cfg['tei_port']}/embed",
            json={"inputs": text},
            timeout=60,
        )
        return np.array(response.json()).reshape(-1)

    def _collection(self):
        return self._weaviate().collections.get(self.cfg["collection"])

    def closest(self, text, gnd_ids, limit):
        """Up to limit of the gnd_ids closest in meaning to text, using the vectors stored in Weaviate."""
        hits = self._collection().query.near_vector(
            near_vector=list(self._embed(unicodedata.normalize("NFC", text))),
            target_vector="label_vector",
            limit=limit * 5,  # one ID can have several entries in the collection
            filters=Filter.by_property("label_id").contains_any(list(gnd_ids)),
        ).objects
        ids = [hit.properties["label_id"] for hit in hits]
        return list(dict.fromkeys(ids))[:limit]

    def map(self, candidate):
        """The closest GND heading to a free candidate: hybrid search (words + vector) in Weaviate.

        score is the hybrid score (shown to the model), cosine_similarity decides whether the hit is accepted."""
        query = unicodedata.normalize("NFC", candidate)
        embedding = self._embed(query)
        hit = self._collection().query.hybrid(
            query=query,
            vector=list(embedding),
            limit=1,
            alpha=self.cfg["alpha"],
            return_metadata=MetadataQuery(score=True),
            include_vector=True,
        ).objects[0]
        return {
            "candidate": candidate,
            "gnd_id": hit.properties["label_id"],
            "term": hit.properties["label_text"],
            "score": round(float(hit.metadata.score), 3),
            "cosine_similarity": float(np.dot(embedding, hit.vector["label_vector"])),
        }
