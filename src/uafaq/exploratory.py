"""Analyses added after the planned runs (see CHANGELOG.md): tuned fusion and other uses of the LLM expansion."""
import json

import numpy as np
from sklearn.preprocessing import normalize

from .config import BOOTSTRAP, CACHE, RESULTS, SEED
from .data import load, read_jsonl
from .experiment import cached_vectors
from .metrics import bootstrap_mean_diff, per_query
from .retrieval import BM25, CharTfidf, ollama_embed, top_k

WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def _cached(name, texts):
    return cached_vectors(CACHE / f"queries_{name}.npz", texts, lambda t: ollama_embed("bge-m3", t))[0]


def _scores_dense(query_vectors, doc_vectors):
    return normalize(query_vectors) @ doc_vectors.T


def _minmax(scores):
    low, high = scores.min(axis=1, keepdims=True), scores.max(axis=1, keepdims=True)
    return (scores - low) / np.where(high > low, high - low, 1)


def _weighted_rrf(rank_a, rank_b, weight, k=60):
    out = []
    for a, b in zip(rank_a, rank_b):
        score = {}
        for w, ranking in ((1 - weight, a), (weight, b)):
            for rank, doc in enumerate(ranking):
                score[doc] = score.get(doc, 0.0) + w / (k + rank + 1)
        out.append(sorted(score, key=score.get, reverse=True)[:100])
    return out


def _ndcg(rankings, queries, ids):
    return np.array([per_query([ids[i] for i in r], set(q["relevant"]))["ndcg10"] for r, q in zip(rankings, queries)])


def run() -> dict:
    corpus, test, dev, _ = load()
    ids = [d["id"] for d in corpus]
    docs = normalize(np.load(CACHE / "docs_bge-m3.npz")["vectors"])
    expansions = {row["id"]: row["answer"] for row in read_jsonl(RESULTS / "expansions.jsonl")}
    queries = test + dev
    q_vec = _cached("plain", [q["text"] for q in queries])
    a_vec = _cached("answer", [expansions[q["id"]] for q in queries])
    n = len(test)

    texts = [d["text"] for d in corpus]
    lexical = {"bm25": BM25().fit(texts), "char": CharTfidf().fit(texts)}
    dense_scores = _scores_dense(q_vec, docs)
    dense_rank = top_k(dense_scores)
    out = {"dense": _ndcg(dense_rank[:n], test, ids).mean()}
    base = _ndcg(dense_rank[:n], test, ids)

    for name, retriever in lexical.items():
        query_matrix = retriever.vectorizer.transform([q["text"] for q in queries])
        if name == "bm25":
            query_matrix.data[:] = 1
        lex_scores = (query_matrix @ retriever.matrix).toarray()
        lex_rank = top_k(lex_scores)
        for method in ("rrf", "convex"):
            def fused(weight):
                if method == "rrf":
                    return _weighted_rrf(lex_rank, dense_rank, weight)
                return top_k(weight * _minmax(dense_scores) + (1 - weight) * _minmax(lex_scores))
            dev_ndcg = {float(w): _ndcg(fused(w)[n:], dev, ids).mean() for w in WEIGHTS}
            best = max(dev_ndcg, key=dev_ndcg.get)
            test_ndcg = _ndcg(fused(best)[:n], test, ids)
            diff, low, high = bootstrap_mean_diff(test_ndcg, base, BOOTSTRAP, SEED)
            out[f"{method}_{name}"] = {"weight_dense": best, "dev": dev_ndcg, "test_ndcg10": float(test_ndcg.mean()),
                                        "d_vs_dense": diff, "ci": [low, high]}

    for name, vectors in (("hyde_answer", a_vec), ("hyde_mean", normalize(q_vec) + normalize(a_vec))):
        ndcg = _ndcg(top_k(_scores_dense(vectors, docs))[:n], test, ids)
        diff, low, high = bootstrap_mean_diff(ndcg, base, BOOTSTRAP, SEED)
        out[name] = {"test_ndcg10": float(ndcg.mean()), "d_vs_dense": diff, "ci": [low, high]}
    out["dense"] = float(out["dense"])
    (RESULTS / "exploratory.json").write_text(json.dumps(out, indent=1))
    return out
