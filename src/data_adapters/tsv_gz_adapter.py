from pathlib import Path

import pandas as pd
import pyarrow

from .base import BaseDataAdapter


class TsvGzTrainAdapter(BaseDataAdapter):
    """Adapter for gzipped TSV train corpora with Arrow index and pref-label map."""

    def __init__(
        self,
        *,
        tsv_gz_file: str,
        index_file: str,
        gnd_pref_labels: str,
        options=None,
        logger=None,
    ) -> None:
        super().__init__(options=options, logger=logger)
        self.tsv_gz_file = Path(tsv_gz_file)
        self.index_file = Path(index_file)
        self.gnd_pref_labels_file = Path(gnd_pref_labels)

    def _validate_inputs(self) -> None:
        for label, path in (
            ("tsv_gz_file", self.tsv_gz_file),
            ("index_file", self.index_file),
            ("gnd_pref_labels", self.gnd_pref_labels_file),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")

    def _load_raw(self) -> pd.DataFrame:
        data_tsv_gz = pd.read_csv(
            self.tsv_gz_file,
            header=None,
            names=["text", "labels"],
            compression="gzip",
            delimiter="\t",
        )
        data_tsv_gz["location"] = data_tsv_gz.index.astype(str)

        data_index = pyarrow.ipc.open_file(self.index_file).read_all().to_pandas()
        pref_labels = pd.read_csv(self.gnd_pref_labels_file)

        expected_index_columns = {"location", "idn"}
        missing_index_columns = expected_index_columns - set(data_index.columns)
        if missing_index_columns:
            raise ValueError(
                f"Index file missing columns: {sorted(missing_index_columns)}"
            )

        expected_pref_columns = {"label_id", "label_text"}
        missing_pref_columns = expected_pref_columns - set(pref_labels.columns)
        if missing_pref_columns:
            raise ValueError(
                f"Pref-label file missing columns: {sorted(missing_pref_columns)}"
            )

        merged = pd.merge(data_tsv_gz, data_index, on=["location"], how="inner")
        self.logger.info("Merged raw train data. shape=%s", merged.shape)
        merged.attrs["pref_labels"] = pref_labels
        return merged

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        pref_labels: pd.DataFrame = raw.attrs["pref_labels"]

        data = raw.copy()
        data["label_list"] = data["labels"].apply(self._parse_label_tokens)

        uri_to_preflabel = dict(
            zip(pref_labels["label_id"], pref_labels["label_text"])
        )
        data["label_texts"] = data["label_list"].apply(
            lambda labels: [uri_to_preflabel[label] for label in labels if label in uri_to_preflabel]
        )

        data = data[["text", "idn", "label_list", "label_texts"]]
        data = data.rename(columns={"idn": "doc_id", "label_list": "label_ids"})
        data["label_texts"] = data["label_texts"].apply(lambda items: "; ".join(items))
        data["label_ids"] = data["label_ids"].apply(lambda items: ", ".join(items))
        self.logger.info("Normalized adapter data. shape=%s", data.shape)
        return data

    @staticmethod
    def _parse_label_tokens(label_field: str) -> list[str]:
        return [token[1:-1].split("/")[-1] for token in label_field.split()]
