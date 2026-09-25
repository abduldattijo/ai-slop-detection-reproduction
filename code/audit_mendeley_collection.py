"""Explore detector flags in the full Mendeley review collection.

This is an exploratory audit, not an authorship classifier with verified
labels. Thresholds are calibrated against the article's 200-source /
400-synthetic training set, then evaluated on the held-out 200 Mendeley
reviews and the 594 remaining distinct source reviews. Source reviews that overlap
the labeled train or test split are excluded from the audit count.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mendeley_source import load_reviews
from detectors import score_all
from run_experiment import METHODS, fpr_at_target_recall, recall_fpr_curve


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Extracted Mendeley collection with ten movie folders")
    parser.add_argument("--train", type=Path,
                        default=Path("data/mendeley_fresh_20260923.jsonl"))
    parser.add_argument("--test", type=Path,
                        default=Path("data/mendeley_test_fresh_20260923.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/mendeley_source_audit_20260925.json"))
    args = parser.parse_args()

    source = load_reviews(args.source_dir)
    train = read_jsonl(args.train)
    test = read_jsonl(args.test)
    train_human = [row for row in train if row.get("label") == "human"]
    synthetic = [row for row in train if row.get("label") == "synthetic"]
    if len(source) != 1000 or len(train_human) != 200 or len(synthetic) != 400 or len(test) != 200:
        raise ValueError(
            f"Unexpected input sizes: source={len(source)}, training humans={len(train_human)}, "
            f"synthetic={len(synthetic)}, test={len(test)}"
        )

    used_texts = {row["text"] for row in train_human + test}
    audit = []
    audit_texts = set()
    for row in source:
        if row["text"] not in used_texts and row["text"] not in audit_texts:
            audit.append(row)
            audit_texts.add(row["text"])
    unique_source_texts = len({row["text"] for row in source})
    if len(source) != 1000 or unique_source_texts != 994 or len(audit) != 594:
        raise ValueError(
            f"Unexpected source deduplication: rows={len(source)}, "
            f"unique_texts={unique_source_texts}, nonoverlap_audit={len(audit)}"
        )

    calibration = train_human + synthetic
    heldout = test
    combined = calibration + heldout + audit
    texts = [row["text"] for row in combined]
    if len(set(texts)) != len(texts):
        raise ValueError("Comparison pool still contains duplicate review texts")
    print(f"Scoring {len(texts)} unique reviews in one shared comparison pool...")
    scores = score_all(texts)

    n_cal = len(calibration)
    n_holdout = len(heldout)
    cal_synthetic = np.array([row.get("label") == "synthetic" for row in calibration])
    result = {
        "purpose": "Exploratory audit only; flags are not verified AI authorship labels.",
        "source_dataset": "Mendeley Data 38j8b6s2mx version 1",
        "counts": {"source_collection_rows": len(source), "source_unique_texts": unique_source_texts,
                   "source_duplicate_rows": len(source) - unique_source_texts,
                   "calibration_source_reference": len(train_human),
                   "calibration_synthetic": len(synthetic), "heldout_source_human_reference": len(heldout),
                   "nonoverlapping_source_audit": len(audit)},
        "methods": {},
        "audit_reviews": [],
    }
    thresholds = {}
    for method in METHODS:
        curve = recall_fpr_curve(scores[method][:n_cal], cal_synthetic)
        point = fpr_at_target_recall(curve, 0.8)
        if point is None:
            raise RuntimeError(f"{method} did not reach 80% recall on calibration data")
        threshold = point["threshold"]
        thresholds[method] = threshold
        holdout_scores = scores[method][n_cal:n_cal + n_holdout]
        audit_scores = scores[method][n_cal + n_holdout:]
        result["methods"][method] = {
            "threshold_for_80pct_synthetic_recall": float(threshold),
            "calibration_source_reference_flagged": int((scores[method][:n_cal][~cal_synthetic] >= threshold).sum()),
            "calibration_source_reference_total": len(train_human),
            "calibration_synthetic_flagged": int((scores[method][:n_cal][cal_synthetic] >= threshold).sum()),
            "calibration_synthetic_total": len(synthetic),
            "heldout_source_reference_flagged": int((holdout_scores >= threshold).sum()),
            "heldout_source_reference_total": len(heldout),
            "audit_source_flagged": int((audit_scores >= threshold).sum()),
            "audit_source_total": len(audit),
        }
        print(f"{method}: cutoff={threshold:.4f}; held-out source {result['methods'][method]['heldout_source_reference_flagged']}/{len(heldout)}; audit flags {result['methods'][method]['audit_source_flagged']}/{len(audit)}")

    for i, row in enumerate(audit):
        j = n_cal + n_holdout + i
        flagged = {method: bool(scores[method][j] >= thresholds[method]) for method in METHODS}
        result["audit_reviews"].append({"id": row["id"], "movie": row["movie"], "sentiment": row["sentiment"],
                                        "flags": flagged,
                                        "combined_score": float(scores["combined_score"][j])})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote audit results to {args.out}")


if __name__ == "__main__":
    main()
