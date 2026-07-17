from argparse import ArgumentParser
import logging
from pathlib import Path

from data_adapters import AdapterOptions, TsvGzTrainAdapter
from data_adapters.constants import REQUIRED_COLUMNS


LOGGER = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Preprocess TSV.GZ corpus data into normalized adapter schema CSV."
    )
    parser.add_argument("--tsv_gz_file", required=True, help="Input gzipped TSV file")
    parser.add_argument("--index_file", required=True, help="Input Arrow index file")
    parser.add_argument(
        "--gnd_pref_labels",
        required=True,
        help="CSV file containing label_id and label_text columns",
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
    adapter = TsvGzTrainAdapter(
        tsv_gz_file=args.tsv_gz_file,
        index_file=args.index_file,
        gnd_pref_labels=args.gnd_pref_labels,
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


if __name__ == "__main__":
    main()
