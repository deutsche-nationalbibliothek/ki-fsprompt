from pathlib import Path

import pandas as pd
import pyarrow

from .base import BaseDataAdapter
from .cleaners import CLEANERS

EMPTY_POLICIES = ("keep", "drop")


class DoclingPartAdapter(BaseDataAdapter):
    """Adapter for Docling-extracted document parts (TOC, abstract, ...).

    One file per document, named ``{idn}{part_suffix}``, sharded into
    subdirectories by the last character of the idn (e.g. ``0/``..``9/``, ``X/``).
    Labels come directly from ground-truth.arrow instead of a pref-label lookup,
    because ground-truth.arrow already carries both the GND-URI and the label text.

    ``part_dir`` and ``part_suffix`` may be lists of equal length to define fallback
    sources (e.g. abstract, else introduction): per document the first source with a
    non-empty file is used. The chosen source is kept in the column ``source``.
    """

    def __init__(
        self,
        *,
        part_dir: str | list[str],
        part_suffix: str | list[str],
        index_file: str,
        ground_truth_file: str,
        kind: str = "title",
        cleaning: str = "none",
        empty_policy: str = "keep",
        options=None,
        logger=None,
    ) -> None:
        super().__init__(options=options, logger=logger)
        part_dirs = [part_dir] if isinstance(part_dir, str) else list(part_dir)
        part_suffixes = [part_suffix] if isinstance(part_suffix, str) else list(part_suffix)
        if len(part_dirs) != len(part_suffixes):
            raise ValueError(
                f"part_dir and part_suffix need the same length, got {len(part_dirs)} and {len(part_suffixes)}"
            )
        # Ordered by priority: the first source with a non-empty file wins.
        self.sources = [(Path(d), s) for d, s in zip(part_dirs, part_suffixes)]
        self.index_file = Path(index_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.kind = kind
        self.cleaning = cleaning
        self.empty_policy = empty_policy

    def _validate_inputs(self) -> None:
        if self.cleaning not in CLEANERS:
            raise ValueError(f"Unknown cleaning '{self.cleaning}'. Options: {sorted(CLEANERS)}")
        if self.empty_policy not in EMPTY_POLICIES:
            raise ValueError(f"Unknown empty_policy '{self.empty_policy}'. Options: {EMPTY_POLICIES}")
        for part_dir, _ in self.sources:
            if not part_dir.is_dir():
                raise FileNotFoundError(f"part_dir does not exist or is not a directory: {part_dir}")
        for label, path in (
            ("index_file", self.index_file),
            ("ground_truth_file", self.ground_truth_file),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")

    @staticmethod
    def _part_path(part_dir: Path, part_suffix: str, idn: str) -> Path:
        shard = idn[-1]
        return part_dir / shard / f"{idn}{part_suffix}"

    def _read_part_text(self, idn: str) -> tuple[str | None, str | None]:
        """Return (text, source name) of the first source with a non-empty file, else (None, None)."""
        for part_dir, part_suffix in self.sources:
            path = self._part_path(part_dir, part_suffix, idn)
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text, part_dir.name
        return None, None

    def _load_raw(self) -> pd.DataFrame:
        index = pyarrow.ipc.open_file(self.index_file).read_all().to_pandas()
        if "idn" not in index.columns:
            raise ValueError("Index file missing column: idn")
        index = index[["idn"]].astype(str)

        ground_truth = pyarrow.ipc.open_file(self.ground_truth_file).read_all().to_pandas()
        ground_truth = ground_truth.astype(str)
        expected_gt_columns = {"idn", "kind", "uri", "label"}
        missing_gt_columns = expected_gt_columns - set(ground_truth.columns)
        if missing_gt_columns:
            raise ValueError(f"Ground truth file missing columns: {sorted(missing_gt_columns)}")

        ground_truth = ground_truth[ground_truth["kind"] == self.kind]

        labels = ground_truth.groupby("idn").agg(
            label_ids=("uri", lambda uris: [uri.split("/")[-1] for uri in uris]),
            label_texts=("label", list),
        )

        texts_and_sources = [self._read_part_text(idn) for idn in index["idn"]]
        raw = index.copy()
        raw["raw_text"] = [text for text, _ in texts_and_sources]
        raw["source"] = [source for _, source in texts_and_sources]

        n_missing = int(raw["source"].isna().sum())
        if n_missing:
            self.logger.warning(
                "%s of %s documents have no non-empty file in any source: %s",
                n_missing, len(index), [f"{d}/*{s}" for d, s in self.sources],
            )
        if len(self.sources) > 1:
            self.logger.info("Documents per source: %s", raw["source"].value_counts().to_dict())

        raw = raw.merge(labels, on="idn", how="left")
        raw["label_ids"] = raw["label_ids"].apply(lambda x: x if isinstance(x, list) else [])
        raw["label_texts"] = raw["label_texts"].apply(lambda x: x if isinstance(x, list) else [])

        self.logger.info("Loaded raw data for parts=%s. shape=%s", [s for _, s in self.sources], raw.shape)
        return raw

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        data = raw.copy()
        data["text"] = data["raw_text"].apply(self._clean_docling_text).apply(CLEANERS[self.cleaning])
        data = data.rename(columns={"idn": "doc_id"})
        data["label_texts"] = data["label_texts"].apply(lambda items: "; ".join(items))
        data["label_ids"] = data["label_ids"].apply(lambda items: ", ".join(items))
        # "source" is no part of the adapter schema; preprocess_selektion.py writes it to a separate file.
        data = data[["text", "doc_id", "label_ids", "label_texts", "source"]]

        empty = data["text"] == ""
        if empty.any():
            self.logger.warning(
                "%s of %s documents have an empty text after cleaning (empty_policy=%s): %s",
                int(empty.sum()), len(data), self.empty_policy, ", ".join(data.loc[empty, "doc_id"]),
            )
            if self.empty_policy == "drop":
                data = data.loc[~empty].reset_index(drop=True)

        self.logger.info("Normalized adapter data. shape=%s", data.shape)
        return data

    @staticmethod
    def _clean_docling_text(raw_text: str | None) -> str:
        """Map missing files to an empty text; the files carry no metadata header, content starts at line 1."""
        # Missing files arrive as NaN, not None, because pandas stores the column as strings.
        if pd.isna(raw_text):
            return ""
        return raw_text.strip()
