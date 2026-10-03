"""Query sample, de-duplicated corpus and nested corpus pools."""
import json
import random

from datasets import load_dataset

from .config import DATA, DATASET, DATASET_REVISION, N_DEV, N_TEST, POOL_SIZES, SEED
from .text import normalise


def _canonical_corpus():
    corpus = load_dataset(DATASET, "ukr-corpus", revision=DATASET_REVISION)["corpus"]
    canonical, texts, alias = {}, {}, {}
    for row in corpus:
        key = " ".join(normalise(row["text"]).split())
        if key not in canonical:
            canonical[key] = row["_id"]
            texts[row["_id"]] = row["text"]
        alias[row["_id"]] = canonical[key]
    return texts, alias


def _queries(split, alias):
    queries = load_dataset(DATASET, "ukr-queries", revision=DATASET_REVISION)[split]
    qrels = load_dataset(DATASET, "ukr-qrels", revision=DATASET_REVISION)[split]
    relevant = {}
    for row in qrels:
        if row["score"] > 0:
            relevant.setdefault(row["query-id"], set()).add(alias[row["corpus-id"]])
    return [{"id": q["_id"], "text": q["text"], "relevant": sorted(relevant[q["_id"]])}
            for q in queries if q["_id"] in relevant]


def build() -> None:
    rng = random.Random(SEED)
    texts, alias = _canonical_corpus()
    test = rng.sample(_queries("test", alias), N_TEST)
    dev = rng.sample(_queries("train", alias), N_DEV)
    positives = sorted({doc for q in test for doc in q["relevant"]})
    others = sorted(set(texts) - set(positives))
    rng.shuffle(others)
    pools = {str(size): positives + others[:size - len(positives)] for size in POOL_SIZES}
    pools["full"] = positives + others

    DATA.mkdir(exist_ok=True)
    with (DATA / "corpus.jsonl").open("w") as handle:
        for doc_id, text in texts.items():
            handle.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    for name, rows in (("test", test), ("dev", dev)):
        with (DATA / f"queries_{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (DATA / "pools.json").write_text(json.dumps(pools))


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def load():
    corpus = read_jsonl(DATA / "corpus.jsonl")
    return (corpus, read_jsonl(DATA / "queries_test.jsonl"), read_jsonl(DATA / "queries_dev.jsonl"),
            json.loads((DATA / "pools.json").read_text()))
