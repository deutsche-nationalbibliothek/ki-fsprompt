from argparse import ArgumentParser
import logging
from pathlib import Path

import pandas as pd
import pyarrow
import pyarrow.ipc

from data_adapters.constants import REQUIRED_COLUMNS
from preprocess_selektion import write_filtered_index


LOGGER = logging.getLogger(__name__)

# Heading per Docling source directory, used when a part comes with a sources CSV
# (written by preprocess_selektion.py --source_output), e.g. abstract with intro fallback.
SOURCE_HEADINGS: dict[str, str] = {
    "Abstract": "Abstract",
    "Intro": "Einleitung",
    "TOC": "Inhaltsverzeichnis",
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Combine several preprocessed selections (e.g. title + abstract) into one normalized CSV."
    )
    parser.add_argument(
        "--part",
        nargs="+",
        action="append",
        required=True,
        metavar="HEADING CSV [SOURCES_CSV]",
        help="One selection in output order: heading shown in the text, normalized CSV, and optionally "
        "the sources CSV of preprocess_selektion.py to take the heading per document from the source.",
    )
    parser.add_argument("--index_file", required=True, help="Input Arrow index file (split assignment and order)")
    parser.add_argument("--csv_output", required=True, help="Output CSV path for normalized schema")
    parser.add_argument(
        "--index_output",
        default=None,
        help="Optional Arrow index containing only the documents present in every part.",
    )
    return parser


def source_heading(source: str, default: str) -> str:
    for prefix, heading in SOURCE_HEADINGS.items():
        if source.startswith(prefix):
            return heading
    return default


def load_part(spec: list[str]) -> tuple[pd.DataFrame, str]:
    """Read one part and return it indexed by doc_id, with the column 'block' = 'Heading:\\ntext'."""
    if len(spec) not in (2, 3):
        raise ValueError(f"--part needs HEADING CSV [SOURCES_CSV], got {spec}")
    heading, csv_path = spec[0], spec[1]
    data = pd.read_csv(csv_path, dtype=str, keep_default_na=False).set_index("doc_id")
    headings = pd.Series(heading, index=data.index)
    if len(spec) == 3:
        sources = pd.read_csv(spec[2], dtype=str).set_index("doc_id")["source"]
        headings = sources.reindex(data.index).fillna(heading).apply(source_heading, default=heading)
    text = data["text"].str.strip()
    data["block"] = (headings + ":\n" + text).where(text != "", "")
    LOGGER.info("Loaded part %s from %s: %s documents", heading, csv_path, len(data))
    return data, heading


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parts = [load_part(spec) for spec in args.part]
    first, _ = parts[0]

    index = pyarrow.ipc.open_file(args.index_file).read_all().column("idn").to_pylist()
    doc_ids = [str(idn) for idn in index]
    # A document is kept only if every part has a non-empty text for it.
    keep = [
        doc_id for doc_id in doc_ids
        if all(doc_id in data.index and data.at[doc_id, "block"] != "" for data, _ in parts)
    ]
    LOGGER.info("%s of %s index documents are present in all parts", len(keep), len(doc_ids))

    # Labels come from ground truth in every part, so they must agree; a mismatch means the inputs drifted apart.
    for data, heading in parts[1:]:
        differs = [
            doc_id for doc_id in keep
            if sorted(data.at[doc_id, "label_ids"].split(", ")) != sorted(first.at[doc_id, "label_ids"].split(", "))
        ]
        if differs:
            raise ValueError(f"label_ids of part {heading} differ for {len(differs)} documents, e.g. {differs[:5]}")

    combined = pd.DataFrame({
        "text": ["\n\n".join(data.at[doc_id, "block"] for data, _ in parts) for doc_id in keep],
        "doc_id": keep,
        "label_ids": [first.at[doc_id, "label_ids"] for doc_id in keep],
        "label_texts": [first.at[doc_id, "label_texts"] for doc_id in keep],
    })[list(REQUIRED_COLUMNS)]

    out_path = Path(args.csv_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    LOGGER.info("Wrote combined output to %s (rows=%s)", out_path, len(combined))

    if args.index_output:
        n_rows = write_filtered_index(args.index_file, set(keep), Path(args.index_output))
        LOGGER.info("Wrote filtered index to %s (rows=%s)", args.index_output, n_rows)


if __name__ == "__main__":
    main()
