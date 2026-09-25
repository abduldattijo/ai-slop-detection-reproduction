"""Score the mixed dataset with each detector and report recall vs.
false-positive-rate at a range of thresholds.

Recall = fraction of synthetic examples caught (score >= threshold).
False-positive rate = fraction of human examples wrongly caught.
This is the "safety" axis the article's thesis rests on, not raw accuracy:
a method with a great catch rate but a high human false-positive rate is
the one to warn readers away from.
"""
import argparse
import json

import numpy as np

from detectors import score_all

METHODS = ["perplexity_score", "near_dup_score", "density_score", "combined_score"]


def load_dataset(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def recall_fpr_curve(scores: np.ndarray, is_synthetic: np.ndarray):
    # Evaluate each distinct score so the operating point is not determined
    # by a coarse threshold grid.
    thresholds = np.concatenate(([np.nextafter(float(scores.max()), np.inf)], np.unique(scores)[::-1]))
    curve = []
    for t in thresholds:
        flagged = scores >= t
        recall = flagged[is_synthetic].mean() if is_synthetic.any() else float("nan")
        fpr = flagged[~is_synthetic].mean() if (~is_synthetic).any() else float("nan")
        curve.append({"threshold": float(t), "recall": float(recall), "false_positive_rate": float(fpr)})
    return curve


def fpr_at_target_recall(curve, target_recall=0.8):
    candidates = [pt for pt in curve if pt["recall"] >= target_recall]
    if not candidates:
        return None
    return min(candidates, key=lambda pt: pt["false_positive_rate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mixed_dataset.jsonl")
    parser.add_argument("--out", default="outputs/results.json")
    args = parser.parse_args()

    rows = load_dataset(args.data)
    texts = [r["text"] for r in rows]
    is_synthetic = np.array([r["label"] == "synthetic" for r in rows])

    print(f"Scoring {len(texts)} examples...")
    scores = score_all(texts)

    results = {"n_total": len(rows), "n_human": int((~is_synthetic).sum()), "n_synthetic": int(is_synthetic.sum())}
    results["methods"] = {}

    for method in METHODS:
        curve = recall_fpr_curve(scores[method], is_synthetic)
        best = fpr_at_target_recall(curve, target_recall=0.8)
        results["methods"][method] = {"curve": curve, "fpr_at_80pct_recall": best}
        if best:
            print(f"{method}: at 80% recall, false-positive rate on human data = {best['false_positive_rate']:.2%}")
        else:
            print(f"{method}: never reaches 80% recall")

    combined = scores["combined_score"]
    human_idx = np.where(~is_synthetic)[0]
    synth_idx = np.where(is_synthetic)[0]

    worst_fp_idx = human_idx[np.argmax(combined[human_idx])]
    worst_fn_idx = synth_idx[np.argmin(combined[synth_idx])]

    results["failure_cases"] = {
        "worst_false_positive": {"id": rows[worst_fp_idx]["id"], "text": rows[worst_fp_idx]["text"], "combined_score": float(combined[worst_fp_idx])},
        "worst_false_negative": {
            "id": rows[worst_fn_idx]["id"],
            "condition": rows[worst_fn_idx]["condition"],
            "text": rows[worst_fn_idx]["text"],
            "combined_score": float(combined[worst_fn_idx]),
        },
    }

    for i, row in enumerate(rows):
        row["combined_score"] = float(combined[i])
    results["per_example_scores"] = [
        {"id": r["id"], "label": r["label"], "condition": r["condition"], "combined_score": r["combined_score"]} for r in rows
    ]

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
