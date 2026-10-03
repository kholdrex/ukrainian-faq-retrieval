"""Retrievers over one corpus pool. Each returns, per query, the top document indices in rank order."""
import json
import time
import urllib.request

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import DEPTH, OLLAMA
from .text import normalise, words


def top_k(scores: np.ndarray, k: int = DEPTH) -> np.ndarray:
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]):
        self.vectorizer = CountVectorizer(analyzer=words)
        tf = self.vectorizer.fit_transform(docs).tocsr().astype(np.float32)
        length = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (tf.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * length / length.mean())
        tf.data = tf.data * (self.k1 + 1) / (tf.data + np.repeat(norm, np.diff(tf.indptr)))
        self.matrix = (tf @ sparse.diags(idf)).T.tocsr()
        return self

    def search(self, queries: list[str]) -> np.ndarray:
        q = self.vectorizer.transform(queries)
        q.data[:] = 1
        return top_k((q @ self.matrix).toarray())


class CharTfidf:
    def fit(self, docs: list[str]):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalise,
                                          sublinear_tf=True, min_df=2, dtype=np.float32)
        self.matrix = self.vectorizer.fit_transform(docs).T.tocsr()
        return self

    def search(self, queries: list[str]) -> np.ndarray:
        return top_k((self.vectorizer.transform(queries) @ self.matrix).toarray())


class Dense:
    def __init__(self, doc_vectors: np.ndarray, encoder):
        self.docs = normalize(doc_vectors)
        self.encoder = encoder

    def search(self, queries: list[str]) -> np.ndarray:
        return top_k(normalize(self.encoder(queries)) @ self.docs.T)

    def search_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return top_k(normalize(vectors) @ self.docs.T)


def rrf(*rankings: np.ndarray, k: int = 60, depth: int = DEPTH) -> np.ndarray:
    fused = []
    for rows in zip(*rankings):
        score = {}
        for ranking in rows:
            for rank, doc in enumerate(ranking):
                score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank + 1)
        fused.append(sorted(score, key=score.get, reverse=True)[:depth])
    return np.array(fused)


def ollama_embed(model: str, texts: list[str], batch: int = 64) -> np.ndarray:
    out = []
    for start in range(0, len(texts), batch):
        body = {"model": model, "input": texts[start:start + batch], "keep_alive": "30m"}
        request = urllib.request.Request(f"{OLLAMA}/api/embed", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            out += json.load(response)["embeddings"]
    return np.asarray(out, dtype=np.float32)


def timed(fn, *args):
    began = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - began
