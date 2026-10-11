"""Single-query latency on the full collection, one query at a time with warm models, on an otherwise idle machine.

Each retriever except BM25 over the expanded query (`bm25_exp`) is timed end to end: query encoding (for dense
retrievers), scoring of all answers and ranking. For the expanded dense variants the time also includes generating the
hypothetical answer; `expansion` times the generation alone.
"""
import json
import time

import numpy as np

from .config import CACHE, DENSE_MODELS, RESULTS, RRF_K
from .data import load
from .expansion import expand
from .experiment import _encoder, cached_vectors
from .retrieval import BM25, CharTfidf, Dense, rrf

N_QUERIES = 100


def _summary(seconds):
    return {"median_ms": 1000 * float(np.median(seconds)), "p95_ms": 1000 * float(np.percentile(seconds, 95)),
            "n": len(seconds)}


def _clock(fn):
    began = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - began


def measure() -> None:
    corpus, test, _, _ = load()
    docs = [d["text"] for d in corpus]
    queries = [q["text"] for q in test[:N_QUERIES]]
    bm25, char = BM25().fit(docs), CharTfidf().fit(docs)
    encoders = {name: _encoder(name) for name in DENSE_MODELS}
    dense = {name: Dense(cached_vectors(CACHE / f"docs_{name}.npz", docs, encoders[name])[0], encoders[name])
             for name in DENSE_MODELS}
    for retriever in dense.values():
        retriever.search(["прогрів"])
    expand(queries[0])

    times = {name: [] for name in ("bm25", "char", *DENSE_MODELS, "rrf", "expansion", "bge-m3_exp", "rrf_exp")}
    for query in queries:
        times["bm25"].append(_clock(lambda: bm25.search([query]))[1])
        times["char"].append(_clock(lambda: char.search([query]))[1])
        for name, retriever in dense.items():
            times[name].append(_clock(lambda: retriever.search([query]))[1])
        times["rrf"].append(_clock(lambda: rrf(bm25.search([query]), dense["bge-m3"].search([query]), k=RRF_K))[1])
        reply, t_expand = _clock(lambda: expand(query))
        times["expansion"].append(t_expand)
        expanded = f"{query}\n{reply['answer']}"
        times["bge-m3_exp"].append(t_expand + _clock(lambda: dense["bge-m3"].search([expanded]))[1])
        times["rrf_exp"].append(t_expand + _clock(
            lambda: rrf(bm25.search([expanded]), dense["bge-m3"].search([expanded]), k=RRF_K))[1])
    (RESULTS / "latency.json").write_text(json.dumps({name: _summary(v) for name, v in times.items()}, indent=1))
