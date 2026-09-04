#!/usr/bin/env python3
"""Predict with an XGBoost model using the same CLI signature as src/R/predict_xgb.r."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.feather as feather


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with XGBoost model from mapped prediction files")

    parser.add_argument(
        "--test_index",
        type=str,
        default="corpora/title/test.arrow",
        help="path to the test index file",
    )
    parser.add_argument(
        "--ranked_predictions",
        type=str,
        default="results/test/ranked_predictions.csv",
        help="path to the predictions file",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="results/xgb_model.rds",
        help="path to the output file",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/test/xgb_predictions.csv",
        help="path to the output file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="number of top k predictions to keep",
    )

    parser.add_argument("in_files", nargs="*", help="mapped_predictions.csv files for single models")
    return parser.parse_args()


def extract_model_name(path: str) -> str:
    raw_name = Path(path).parent.name
    return re.sub(r"[/-]", "_", raw_name)


def load_test_index(path: str) -> pd.DataFrame:
    test_index = feather.read_feather(path)
    return test_index.loc[:, ["idn"]].rename(columns={"idn": "doc_id"}).assign(
        doc_id=lambda d: d["doc_id"].astype(str)
    )


def load_ranked_predictions(path: str) -> pd.DataFrame:
    ranked_predictions = pd.read_csv(
        path,
        usecols=["doc_id", "label_id", "score"],
        dtype={"doc_id": "string", "label_id": "string", "score": "float64"},
    )
    return ranked_predictions.rename(columns={"score": "score_relevance"})


def load_single_model_predictions(in_files: Sequence[str]) -> list[pd.DataFrame]:
    results: list[pd.DataFrame] = []
    for in_file in in_files:
        model_name = extract_model_name(in_file)
        frame = pd.read_csv(
            in_file,
            usecols=["doc_id", "label_id", "score"],
            dtype={"doc_id": "string", "label_id": "string", "score": "float64"},
        ).rename(columns={"score": f"score_{model_name}"})
        results.append(frame)
    return results


def merge_predictions(single_model_preds: list[pd.DataFrame]) -> pd.DataFrame:
    merged = single_model_preds[0]
    for frame in single_model_preds[1:]:
        merged = merged.merge(frame, how="outer", on=["doc_id", "label_id"])
    return merged


def ensure_parent(path_str: str) -> None:
    path = Path(path_str)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    if not args.in_files:
        raise ValueError("At least one mapped_predictions.csv input file must be provided.")

    print(args.in_files)
    print(f"Current working dir: {Path.cwd()}")

    print(f"Reading training index from {args.test_index}...")
    test_index = load_test_index(args.test_index)

    print(f"Reading ranked predictions from {args.ranked_predictions}...")
    ranked_predictions = load_ranked_predictions(args.ranked_predictions)

    print("Reading single model predictions...")
    single_model_preds = load_single_model_predictions(args.in_files)

    print("Joining predictions on doc_id and label_id...")
    predictions_ensemble = merge_predictions(single_model_preds)

    drop_cols = [c for c in predictions_ensemble.columns if c.startswith("max_cosine_similarity")]
    if drop_cols:
        predictions_ensemble = predictions_ensemble.drop(columns=drop_cols)
    predictions_ensemble = predictions_ensemble.fillna(0)

    joined_predictions = predictions_ensemble.merge(
        ranked_predictions,
        how="outer",
        on=["doc_id", "label_id"],
    )

    na_relevance = int(joined_predictions["score_relevance"].isna().sum())
    print(f"Number of rows with NA relevance score: {na_relevance}")

    mean_relevance = joined_predictions["score_relevance"].mean(skipna=True)
    joined_predictions["score_relevance"] = joined_predictions["score_relevance"].fillna(mean_relevance)

    score_cols = [c for c in joined_predictions.columns if "score" in c]
    joined_predictions[score_cols] = joined_predictions[score_cols].fillna(0.0)

    with open(args.model_file, "rb") as f:
        model = pickle.load(f)

    print("Making predictions...")
    preds = model.predict_proba(joined_predictions[score_cols])[:, 1]

    output = joined_predictions.assign(score=preds).loc[:, ["doc_id", "label_id", "score"]]
    output = (
        output.sort_values(["doc_id", "score"], ascending=[True, False])
        .groupby("doc_id", as_index=False)
        .head(args.top_k)
        .reset_index(drop=True)
    )

    output = output.loc[output["doc_id"].isin(test_index["doc_id"])]

    ensure_parent(args.out_csv)
    output.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
