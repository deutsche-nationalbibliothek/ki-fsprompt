# parse prefLabels from .ttl file

import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import pyoxigraph
from pyoxigraph import RdfFormat
import pandas as pd
import unicodedata
import logging

PREF_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#altLabel"

def parse_vocab(
    ttl_path: Path, use_altLabels: bool = True, phrase: str = None
) -> pd.DataFrame:
    logging.info(f"Parsing vocabulary from {ttl_path}")
    with ttl_path.open("rb") as f:
        graph = pyoxigraph.parse(f, RdfFormat.TURTLE)
        labels: list[(str, str, bool)] = []

        for s, p, o, _ in tqdm(graph, desc="Processing triples"):
            uri = s.value
            label_id = uri.split("/")[-1]  # extract label_id from uri
            is_prefLabel = p.value == PREF_LABEL_IRI
            is_altLabel = p.value == ALT_LABEL_IRI
            label_text = o.value if phrase is None else f"{phrase}{o.value}"
            label_text = unicodedata.normalize("NFC", label_text)  # normalize unicode
            if is_prefLabel:
                labels.append((label_id, label_text, True))
            elif is_altLabel and use_altLabels:
                labels.append((label_id, label_text, False))

        labels = pd.DataFrame(
            labels, columns=["label_id", "label_text", "is_prefLabel"]
        )

        return labels

def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl_file", help="File containing the TTL data")
    parser.add_argument("--csv_output", help="Output CSV file for the extracted prefLabels")
    args = parser.parse_args()

    ttl_file = Path(args.ttl_file)
    csv_output = args.csv_output

    df = parse_vocab(ttl_file, use_altLabels=False, phrase=None)
    df.to_csv(csv_output, index=False, columns=["label_id", "label_text"])

execute()