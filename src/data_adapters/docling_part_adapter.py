from pathlib import Path

import pandas as pd
import pyarrow

from .base import BaseDataAdapter
from .cleaners import CLEANERS


class DoclingPartAdapter(BaseDataAdapter):
    """Adapter for a single Docling-extracted document part (TOC, abstract, ...).

    One file per document, named ``{idn}{part_suffix}``, sharded into
    subdirectories by the last character of the idn (e.g. ``0/``..``9/``, ``X/``).
    Labels come directly from ground-truth.arrow instead of a pref-label lookup,
    because ground-truth.arrow already carries both the GND-URI and the label text.
    """

    def __init__(
        self,
        *,
        part_dir: str,
        part_suffix: str,
        index_file: str,
        ground_truth_file: str,
        kind: str = "title",
        cleaning: str = "none",
        options=None,
        logger=None,
    ) -> None:
        super().__init__(options=options, logger=logger)
        self.part_dir = Path(part_dir)
        self.part_suffix = part_suffix
        self.index_file = Path(index_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.kind = kind
        self.cleaning = cleaning

    def _validate_inputs(self) -> None:
        if self.cleaning not in CLEANERS:
            raise ValueError(f"Unknown cleaning '{self.cleaning}'. Options: {sorted(CLEANERS)}")
        if not self.part_dir.is_dir():
            raise FileNotFoundError(f"part_dir does not exist or is not a directory: {self.part_dir}")
        for label, path in (
            ("index_file", self.index_file),
            ("ground_truth_file", self.ground_truth_file),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")

    def _part_path(self, idn: str) -> Path:
        shard = idn[-1]
        return self.part_dir / shard / f"{idn}{self.part_suffix}"

    def _read_part_text(self, idn: str) -> str | None:
        path = self._part_path(idn)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

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

        raw_texts = index["idn"].apply(self._read_part_text)
        n_missing = raw_texts.isna().sum()
        if n_missing:
            self.logger.warning(
                "%s of %s documents have no %s file under %s",
                n_missing, len(index), self.part_suffix, self.part_dir,
            )

        raw = index.copy()
        raw["raw_text"] = raw_texts
        raw = raw.merge(labels, on="idn", how="left")
        raw["label_ids"] = raw["label_ids"].apply(lambda x: x if isinstance(x, list) else [])
        raw["label_texts"] = raw["label_texts"].apply(lambda x: x if isinstance(x, list) else [])

        self.logger.info("Loaded raw data for part=%s. shape=%s", self.part_suffix, raw.shape)
        return raw

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        data = raw.copy()
        data["text"] = data["raw_text"].apply(self._clean_docling_text).apply(CLEANERS[self.cleaning])
        data = data.rename(columns={"idn": "doc_id"})
        data["label_texts"] = data["label_texts"].apply(lambda items: "; ".join(items))
        data["label_ids"] = data["label_ids"].apply(lambda items: ", ".join(items))
        data = data[["text", "doc_id", "label_ids", "label_texts"]]

        n_empty = (data["text"] == "").sum()
        if n_empty:
            self.logger.warning("%s of %s documents have an empty text after cleaning", n_empty, len(data))

        self.logger.info("Normalized adapter data. shape=%s", data.shape)
        return data

    @staticmethod
    def _clean_docling_text(raw_text: str | None) -> str:
        """Strip the Docling metadata header (lines starting with '#' or '>')."""
        if raw_text is None:
            return ""
        lines = [
            line for line in raw_text.splitlines()
            if not line.lstrip().startswith(("#", ">"))
        ]
        return "\n".join(lines).strip()
