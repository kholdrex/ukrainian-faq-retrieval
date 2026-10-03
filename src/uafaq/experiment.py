"""Runs every retriever on every corpus pool and stores rankings, per-query metrics and timings."""
import json

import numpy as np

from .config import CACHE, DENSE_MODELS, RESULTS, RRF_K
from .data import load, read_jsonl
from .expansion import expand
from .metrics import per_query
from .retrieval import BM25, CharTfidf, Dense, ollama_embed, rrf, timed

EXPANSIONS = RESULTS / "expansions.jsonl"


def _encoder(name):
    kind, model = DENSE_MODELS[name]
    if kind == "ollama":
        return lambda texts: ollama_embed(model, texts)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model, device="mps")
    return lambda texts: st.encode(texts, batch_size=64, convert_to_numpy=True)


def doc_vectors(name, corpus, encoder):
    path = CACHE / f"docs_{name}.npy"
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors, seconds = timed(encoder, [d["text"] for d in corpus])
    np.save(path, vectors)
    _append(RESULTS / "timings.jsonl", {"step": "embed_corpus", "model": name, "docs": len(corpus),
                                        "seconds": seconds})
    return vectors


def expansions(queries):
    done = {row["id"]: row for row in read_jsonl(EXPANSIONS)} if EXPANSIONS.exists() else {}
    for query in queries:
        if query["id"] not in done:
            done[query["id"]] = {"id": query["id"], **expand(query["text"])}
            _append(EXPANSIONS, done[query["id"]])
    return [done[q["id"]] for q in queries]


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run() -> None:
    corpus, test, dev, pools = load()
    position = {d["id"]: i for i, d in enumerate(corpus)}
    queries = test + dev
    texts = [q["text"] for q in queries]
    expanded = [f"{q['text']}\n{e['answer']}" for q, e in zip(queries, expansions(queries))]

    encoders = {name: _encoder(name) for name in DENSE_MODELS}
    vectors = {name: doc_vectors(name, corpus, enc) for name, enc in encoders.items()}
    query_vectors = {name: enc(texts) for name, enc in encoders.items()}
    expanded_vectors = encoders["bge-m3"](expanded)

    rows = []
    for pool_name, doc_ids in pools.items():
        index = np.array([position[d] for d in doc_ids])
        docs = [corpus[i]["text"] for i in index]
        subset = queries if pool_name == "full" else test
        n = len(subset)
        bm25, build_bm25 = timed(BM25().fit, docs)
        char, build_char = timed(CharTfidf().fit, docs)
        runs = {
            "bm25": bm25.search(texts[:n]),
            "char": char.search(texts[:n]),
            "bm25_exp": bm25.search(expanded[:n]),
        }
        for name in DENSE_MODELS:
            runs[name] = Dense(vectors[name][index], None).search_vectors(query_vectors[name][:n])
        runs["bge-m3_exp"] = Dense(vectors["bge-m3"][index], None).search_vectors(expanded_vectors[:n])
        runs["rrf"] = rrf(runs["bm25"], runs["bge-m3"], k=RRF_K)
        runs["rrf_exp"] = rrf(runs["bm25_exp"], runs["bge-m3_exp"], k=RRF_K)
        _append(RESULTS / "timings.jsonl", {"step": "build_lexical", "pool": pool_name, "docs": len(docs),
                                            "bm25_s": build_bm25, "char_s": build_char})
        for system, ranking in runs.items():
            for number, (query, ranked) in enumerate(zip(subset, ranking)):
                ids = [doc_ids[i] for i in ranked]
                rows.append({"pool": pool_name, "system": system, "query": query["id"],
                             "split": "test" if number < len(test) else "dev",
                             **per_query(ids, set(query["relevant"]))})
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "per_query.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
