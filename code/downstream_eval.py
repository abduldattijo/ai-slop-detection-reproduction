"""Train the same sentiment classifier on four training-set variants and
compare accuracy on a held-out Mendeley source review test set.

Variants:
  - untouched:    full mixed set, no cleaning
  - filtered:     hard-remove anything at/above the combined-score threshold
                   that hits 80% recall at the lowest source-reference flag
                   rate (see run_experiment.py)
  - downweighted: keep everything, sample_weight = 1 - combined_score
  - oracle:       source-only comparison reference

The comparison measures what filtering did in this particular run.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from detectors import embed_texts

LABEL_MAP = {"positive": 1, "negative": 0}


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_held_out_test(n_test: int, seed: int = 1):
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="test")
    idx = list(range(len(ds)))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_pos, n_neg = (n_test + 1) // 2, n_test // 2
    pos, neg = [], []
    for i in idx:
        row = ds[i]
        bucket, target = (pos, n_pos) if row["label"] == 1 else (neg, n_neg)
        if len(bucket) < target:
            bucket.append(row["text"])
        if len(pos) >= n_pos and len(neg) >= n_neg:
            break
    texts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return texts, labels


def load_test_jsonl(path: str):
    rows = load_jsonl(path)
    texts = [r["text"] for r in rows]
    labels = [LABEL_MAP[r["sentiment"]] for r in rows]
    return texts, labels


def build_variants(rows, results):
    threshold_info = results["methods"]["combined_score"]["fpr_at_80pct_recall"]
    threshold = threshold_info["threshold"] if threshold_info else 1.0

    scores_by_id = {r["id"]: r["combined_score"] for r in results["per_example_scores"]}

    untouched = rows
    filtered = [r for r in rows if scores_by_id[r["id"]] < threshold]
    oracle = [r for r in rows if r["label"] == "human"]

    return {
        "untouched": {"rows": untouched, "weights": None},
        "filtered": {"rows": filtered, "weights": None},
        "downweighted": {"rows": rows, "weights": [1.0 - scores_by_id[r["id"]] for r in rows]},
        "oracle": {"rows": oracle, "weights": None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mixed_dataset.jsonl")
    parser.add_argument("--results", default="outputs/results.json")
    parser.add_argument("--test-data", help="Optional JSONL of a separate held-out source review test set")
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--out", default="outputs/downstream_results.json")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    with open(args.results) as f:
        results = json.load(f)

    if args.test_data:
        print(f"Loading held-out source review examples from {args.test_data}...")
        test_texts, test_labels = load_test_jsonl(args.test_data)
    else:
        print(f"Pulling {args.n_test} held-out source review examples...")
        test_texts, test_labels = load_held_out_test(args.n_test)
    test_embeddings = embed_texts(test_texts)

    variants = build_variants(rows, results)

    summary = {}
    for name, variant in variants.items():
        train_rows = variant["rows"]
        train_texts = [r["text"] for r in train_rows]
        train_labels = [LABEL_MAP[r["sentiment"]] for r in train_rows]
        train_embeddings = embed_texts(train_texts)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_embeddings, train_labels, sample_weight=variant["weights"])

        preds = clf.predict(test_embeddings)
        acc = accuracy_score(test_labels, preds)
        summary[name] = {"n_train": len(train_rows), "test_accuracy": acc}
        print(f"{name}: n_train={len(train_rows)}, test_accuracy={acc:.3f}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote downstream results to {args.out}")


if __name__ == "__main__":
    main()
