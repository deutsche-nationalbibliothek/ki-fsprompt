from argparse import ArgumentParser
import logging
from pathlib import Path

import pyarrow
import pyarrow.ipc

from data_adapters import AdapterOptions, DoclingPartAdapter
from data_adapters.cleaners import CLEANERS
from data_adapters.constants import REQUIRED_COLUMNS
from data_adapters.docling_part_adapter import EMPTY_POLICIES


LOGGER = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Preprocess a single Docling-extracted document part (e.g. TOC) into normalized adapter schema CSV."
    )
    parser.add_argument(
        "--part_dir",
        required=True,
        help="Directory with one file per document, sharded by last idn character (e.g. corpora/TOC_Extraction)",
    )
    parser.add_argument(
        "--part_suffix",
        required=True,
        help="Filename suffix after the idn (e.g. _TOC.md)",
    )
    parser.add_argument("--index_file", required=True, help="Input Arrow index file (split assignment)")
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="ground-truth.arrow with columns idn, kind, uri, label",
    )
    parser.add_argument(
        "--kind",
        default="title",
        help="Which 'kind' of ground-truth labels to use (default: title)",
    )
    parser.add_argument(
        "--cleaning",
        default="none",
        choices=sorted(CLEANERS),
        help="Part-specific text cleaning applied after header removal (default: none)",
    )
    parser.add_argument(
        "--empty_policy",
        default="keep",
        choices=EMPTY_POLICIES,
        help="What to do with documents whose text is empty after cleaning (default: keep)",
    )
    parser.add_argument(
        "--index_output",
        default=None,
        help="Optional Arrow index containing only the documents written to --csv_output. "
        "Needed for evaluation when --empty_policy drop removes documents.",
    )
    parser.add_argument(
        "--csv_output",
        required=True,
        help="Output CSV path for normalized schema",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional limit for rows after normalization",
    )
    parser.add_argument(
        "--debug_rows",
        type=int,
        default=None,
        help="Optional debug row cap applied after max_rows",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Enforce non-null checks on required columns",
    )
    parser.add_argument(
        "--no_strict",
        dest="strict",
        action="store_false",
        help="Disable non-null checks on required columns",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    options = AdapterOptions(
        max_rows=args.max_rows,
        debug_rows=args.debug_rows,
        strict=args.strict,
    )
    adapter = DoclingPartAdapter(
        part_dir=args.part_dir,
        part_suffix=args.part_suffix,
        index_file=args.index_file,
        ground_truth_file=args.ground_truth,
        kind=args.kind,
        cleaning=args.cleaning,
        empty_policy=args.empty_policy,
        options=options,
        logger=LOGGER,
    )

    normalized = adapter.load()
    output_columns = list(REQUIRED_COLUMNS)
    normalized = normalized[output_columns]

    out_path = Path(args.csv_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(out_path, index=False)

    LOGGER.info("Wrote preprocessed output to %s", out_path)
    LOGGER.info("Rows=%s Columns=%s", len(normalized), output_columns)

    if args.index_output:
        n_rows = write_filtered_index(args.index_file, set(normalized["doc_id"]), Path(args.index_output))
        LOGGER.info("Wrote filtered index to %s (rows=%s)", args.index_output, n_rows)


def write_filtered_index(index_file: str, doc_ids: set[str], out_path: Path) -> int:
    """Copy the split index, keeping only rows whose idn is in doc_ids (original order preserved)."""
    index = pyarrow.ipc.open_file(index_file).read_all()
    # pyarrow cannot filter string_view columns (as written by polars), so cast them to string.
    index = index.cast(pyarrow.schema([
        pyarrow.field(f.name, pyarrow.string() if pyarrow.types.is_string_view(f.type) else f.type)
        for f in index.schema
    ]))
    keep = pyarrow.array([str(idn) in doc_ids for idn in index.column("idn").to_pylist()])
    filtered = index.filter(keep)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pyarrow.ipc.new_file(out_path, filtered.schema) as writer:
        writer.write_table(filtered)
    return filtered.num_rows


if __name__ == "__main__":
    main()
