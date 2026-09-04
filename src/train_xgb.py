#!/usr/bin/env python3
"""
Train an XGBoost relevance model that combines predictions from multiple single LLMs.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import pickle
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.feather as feather
import xgboost as xgb


def parse_bool(value: str) -> bool:
    """Parse R-style logical argument values."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid logical value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost model from mapped prediction files")

    parser.add_argument(
        "--ground_truth",
        type=str,
        default="../../corpora/ground-truth.arrow",
        help="path to the ground truth file",
    )
    parser.add_argument(
        "--train_index",
        type=str,
        default="../../corpora/title/test.arrow",
        help="path to the test index file",
    )
    parser.add_argument(
        "--ranked_predictions",
        type=str,
        default="ranked_predictions.csv",
        help="path to the predictions file",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=100,
        help="number of boosting iterations",
    )
    parser.add_argument(
        "--interaction_depth",
        type=int,
        default=4,
        help="maximum depth of interaction in the GBM model",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.2,
        help="shrinkage parameter for the GBM model between 0 and 1",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.5,
        help="subsample ratio of the training instances",
    )
    parser.add_argument(
        "--verbose",
        type=parse_bool,
        default=False,
        help="whether to print verbose output during training",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="results/xgb_model.pkl",
        help="path to the output file",
    )
    parser.add_argument(
        "--importance_plot",
        type=str,
        default="results/feature_importance.png",
        help="path to the output plot file",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=20,
        help="number of jobs to run in parallel",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="title",
        help="ft or title",
    )

    parser.add_argument("in_files", nargs="*", help="mapped_predictions.csv files for single models")
    return parser.parse_args()


def extract_model_name(path: str) -> str:
    raw_name = Path(path).parent.name
    return re.sub(r"[/-]", "_", raw_name)


def load_ground_truth(path: str, kind: str) -> pd.DataFrame:
    ground_truth = feather.read_feather(path)
    ground_truth = ground_truth.loc[ground_truth["kind"] == kind, ["idn", "uri"]].copy()
    ground_truth["doc_id"] = ground_truth["idn"].astype(str)
    ground_truth["label_id"] = (
        ground_truth["uri"].astype(str).str.extract(r"([0-9X]{9,10})", expand=False)
    )
    return ground_truth.loc[:, ["doc_id", "label_id"]].dropna()


def load_train_index(path: str) -> pd.DataFrame:
    train_index = feather.read_feather(path)
    return train_index.loc[:, ["idn"]].rename(columns={"idn": "doc_id"}).assign(
        doc_id=lambda d: d["doc_id"].astype(str)
    ).drop_duplicates()


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


def save_importance_plot(model: xgb.XGBClassifier, output_path: str) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    xgb.plot_importance(model, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.in_files:
        raise ValueError("At least one mapped_predictions.csv input file must be provided.")

    print(args.in_files)
    print(f"Reading ground truth from {args.ground_truth}...")
    ground_truth = load_ground_truth(args.ground_truth, args.kind)

    print(f"Reading training index from {args.train_index}...")
    train_index = load_train_index(args.train_index)

    print(f"Reading ranked predictions from {args.ranked_predictions}...")
    ranked_predictions = load_ranked_predictions(args.ranked_predictions)

    print("Reading single model predictions...")
    single_model_preds = load_single_model_predictions(args.in_files)

    print("Joining predictions on doc_id and label_id...")
    predictions_ensemble = merge_predictions(single_model_preds)

    joined_predictions = predictions_ensemble.merge(
        ranked_predictions,
        how="outer",
        on=["doc_id", "label_id"],
    )
    joined_predictions = joined_predictions.loc[joined_predictions["score_relevance"].notna()].copy()

    gold_standard = ground_truth.merge(train_index, how="inner", on="doc_id")
    predicted = joined_predictions.merge(train_index, how="inner", on="doc_id")

    gold_pairs = gold_standard.assign(gold=1).loc[:, ["doc_id", "label_id", "gold"]]
    comp_train = predicted.merge(gold_pairs, how="left", on=["doc_id", "label_id"])
    comp_train["gold"] = comp_train["gold"].fillna(0).astype(int)

    score_cols = [c for c in comp_train.columns if "score" in c]
    comp_train[score_cols] = comp_train[score_cols].fillna(0.0)

    if not score_cols:
        raise ValueError("No score feature columns found for training.")

    X = comp_train[score_cols]
    y = comp_train["gold"]

    model = xgb.XGBClassifier(
        n_estimators=args.n_rounds,
        max_depth=args.interaction_depth,
        learning_rate=args.shrinkage,
        subsample=args.subsample,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=args.n_jobs,
        random_state=4957238,
        verbosity=1 if args.verbose else 0,
    )

    model.fit(X, y)

    save_importance_plot(model, args.importance_plot)

    ensure_parent(args.model_file)
    with open(args.model_file, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
