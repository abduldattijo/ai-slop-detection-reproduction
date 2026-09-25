# AI-written review detection experiment

This package contains the exact review files and scripts used for the experiment described in the accompanying article. It lets you rerun the detector comparison, the sentiment-classifier evaluation, and (if you download the source archive) the Mendeley collection audit on your own computer.

## Requirements

- Python 3.11 or newer
- The packages in `requirements.txt`
- About 2 GB of free disk space for the first run, when the pretrained models are downloaded

The first run downloads GPT-2 and the `all-MiniLM-L6-v2` sentence-embedding model from Hugging Face. The detector and classifier then run locally. The included review files preserve the exact generated examples used in the reported run, so they can be used directly in the analysis.

## Run the experiment

Clone the repository, enter its directory, create and activate a virtual environment, then install the dependencies:

```bash
git clone https://github.com/abduldattijo/ai-slop-detection-reproduction.git
cd ai-slop-detection-reproduction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p outputs
```

Score the 600 training reviews with all four detector scores (perplexity, near-duplicate similarity, embedding density, and their combined score):

```bash
python code/run_experiment.py \
  --data data/reviews_train.jsonl \
  --out outputs/detector_results.json
```

Train and evaluate the sentiment classifier using the same four data treatments reported in the article:

```bash
python code/downstream_eval.py \
  --data data/reviews_train.jsonl \
  --results outputs/detector_results.json \
  --test-data data/reviews_test.jsonl \
  --out outputs/downstream_results.json
```

The test file contains 200 source reviews kept separate from training. The scripts write their results under `outputs/`. Compare them with the captured run in `reference_outputs/`. Small numerical differences can occur across operating systems, hardware, or library versions.

## Repeat the archive audit

Download version 1 of the source collection from [Mendeley Data](https://data.mendeley.com/datasets/38j8b6s2mx/1). Extract `Dataset.rar` so the extracted directory contains the ten movie folders, then run:

```bash
python code/audit_mendeley_collection.py \
  --source-dir /path/to/extracted/Dataset \
  --train data/reviews_train.jsonl \
  --test data/reviews_test.jsonl \
  --out outputs/source_audit.json
```

The Mendeley reviews are treated as human-written because the collection says they were extracted from IMDb and was published in 2019, before [ChatGPT launched publicly](https://openai.com/index/chatgpt/). The collection does not verify each review's author. The audit reports detector flags, not confirmed AI authorship. Attribution and license details are in `DATASET-ATTRIBUTION.md`.

## Files

- `data/reviews_train.jsonl`: 200 source reviews and 400 generated reviews used to fit and score the detectors and train the classifier.
- `data/reviews_test.jsonl`: 200 source reviews reserved for classifier evaluation.
- `code/`: the detector, evaluation, and archive-audit scripts.
- `reference_outputs/`: the saved JSON results used for the article's reported figures and counts.

The exact generated reviews used in the reported run are included in the training file. This package does not regenerate them: doing so requires a model API key, access to the same model versions, and would still produce different text and potentially different results.
