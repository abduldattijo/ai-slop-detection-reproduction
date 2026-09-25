"""Detection signals for the AI-slop experiment.

Three cheap, no-paid-API signals:
  - perplexity: under a small open causal LM (GPT-2)
  - near_dup: max pairwise cosine similarity in embedding space
  - density: mean distance to k nearest neighbors in embedding space
    (AI text tends to sit in denser neighborhoods than diverse human writing)

Each returns one float per input text; higher = more suspicious, except
perplexity, which is inverted (low perplexity = more suspicious/templated)
before being folded into a combined score.
"""
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer

_PPL_MODEL_NAME = "gpt2"
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_ppl_tokenizer = None
_ppl_model = None
_embed_model = None


def _load_ppl_model():
    global _ppl_tokenizer, _ppl_model
    if _ppl_model is None:
        _ppl_tokenizer = AutoTokenizer.from_pretrained(_PPL_MODEL_NAME)
        _ppl_model = AutoModelForCausalLM.from_pretrained(_PPL_MODEL_NAME)
        _ppl_model.eval()
    return _ppl_tokenizer, _ppl_model


def _load_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def compute_perplexity(texts: list[str]) -> np.ndarray:
    tokenizer, model = _load_ppl_model()
    scores = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if enc["input_ids"].shape[1] < 2:
                scores.append(float("nan"))
                continue
            out = model(**enc, labels=enc["input_ids"])
            scores.append(torch.exp(out.loss).item())
    return np.array(scores)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _load_embed_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def near_dup_score(embeddings: np.ndarray) -> np.ndarray:
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, -1.0)
    return sims.max(axis=1)


def density_score(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    n_neighbors = min(k + 1, len(embeddings))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)
    mean_dist = dists[:, 1:].mean(axis=1)
    return -mean_dist


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=np.nanmean(x))
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def score_all(texts: list[str]) -> dict[str, np.ndarray]:
    """Returns per-signal scores plus a combined score, all normalized to
    [0, 1] where higher = more AI-like."""
    ppl = compute_perplexity(texts)
    embeddings = embed_texts(texts)
    near_dup = near_dup_score(embeddings)
    density = density_score(embeddings)

    inv_ppl = _normalize(-ppl)
    near_dup_n = _normalize(near_dup)
    density_n = _normalize(density)
    combined = (inv_ppl + near_dup_n + density_n) / 3.0

    return {
        "perplexity": ppl,
        "perplexity_score": inv_ppl,
        "near_dup_score": near_dup_n,
        "density_score": density_n,
        "combined_score": combined,
        "embeddings": embeddings,
    }
