#!/usr/bin/env python
"""Vergleicht Bedingungen auf Ebene der gemappten Kandidaten.

Zeigt, wieviel eines Metrik-Unterschieds echter Qualitaetsverlust ist und wieviel
nur daran liegt, dass eine Bedingung mehr Kandidaten erzeugt, als @k beruecksichtigt.

    python src/compare_conditions.py --k 5 \
        Titel=pipelines/train/results/.../mapped_predictions.csv \
        TOC=pipelines/train/predictions_model_wise/llama-3B/mapped_predictions.csv
"""
import argparse
import numpy as np
import pandas as pd
import pyarrow.ipc


def load_gold(ground_truth: str, kind: str) -> dict[str, set[str]]:
    gt = pyarrow.ipc.open_file(ground_truth).read_all().to_pandas().astype(str)
    gt = gt[gt["kind"] == kind]
    return gt.groupby("idn")["uri"].apply(
        lambda uris: {u.split("/")[-1] for u in uris}
    ).to_dict()


def stats(path: str, gold: dict[str, set[str]], k: int) -> dict[str, float]:
    data = pd.read_csv(path, dtype={"doc_id": str, "label_id": str})
    rows = []
    for doc, group in data.groupby("doc_id"):
        truth = gold.get(doc)
        if not truth:
            continue
        ids = list(group.sort_values("score", ascending=False)["label_id"])
        rows.append((
            len(ids),
            len(truth & set(ids[:k])) / len(truth),
            len(truth & set(ids)) / len(truth),
        ))
    a = np.array(rows, dtype=float)
    return {"docs": len(a), "cand": a[:, 0].mean(),
            "rec_at_k": a[:, 1].mean(), "rec_all": a[:, 2].mean()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("conditions", nargs="+", metavar="NAME=PFAD",
                   help="Benannte mapped_predictions.csv, mind. eine")
    p.add_argument("--ground_truth", default="corpora/ground-truth.arrow")
    p.add_argument("--kind", default="title")
    p.add_argument("--k", type=int, default=5, help="Metrik-Tiefe (default: 5)")
    args = p.parse_args()

    gold = load_gold(args.ground_truth, args.kind)
    k = args.k
    results = {}
    print(f"{'Bedingung':<22s} {'Dok':>5s} {'Kand./Dok':>10s} "
          f"{'rec@'+str(k):>8s} {'rec@ALL':>9s} {'durch @'+str(k)+' verloren':>18s}")
    for item in args.conditions:
        name, _, path = item.partition("=")
        s = stats(path, gold, k)
        results[name] = s
        print(f"{name:<22s} {s['docs']:5d} {s['cand']:10.1f} "
              f"{s['rec_at_k']:8.4f} {s['rec_all']:9.4f} {s['rec_all']-s['rec_at_k']:18.4f}")

    names = list(results)
    if len(names) == 2:
        a, b = (results[n] for n in names)
        total = b["rec_at_k"] - a["rec_at_k"]
        real = b["rec_all"] - a["rec_all"]
        cut = (b["rec_all"] - b["rec_at_k"]) - (a["rec_all"] - a["rec_at_k"])
        print(f"\nZerlegung von {names[1]} gegen {names[0]}:")
        print(f"  Gesamtdifferenz rec@{k}   {total:+.4f}")
        share = lambda x: f"{abs(x)/abs(total):.0%}" if total else "n/a"
        print(f"  davon echte Qualitaet    {real:+.4f}  ({share(real)})")
        print(f"  davon Abschneiden bei {k}  {-cut:+.4f}  ({share(cut)})")


if __name__ == "__main__":
    main()
