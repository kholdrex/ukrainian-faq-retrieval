# Retrieval of Ukrainian FAQ answers

Code, generated query expansions and analysis for the paper *Lexical, dense, hybrid and LLM-expanded retrieval of
Ukrainian FAQ answers as the collection grows* (O. Kholodniak).

On the Ukrainian part of WebFAQ (1,000 test questions; collections of 2,000, 10,000 and 67,721 answers) the following
retrievers are compared (`retrieval.py`, `experiment.py`):

| Retriever | Description |
|---|---|
| `bm25` | word BM25 (k1 = 1.2, b = 0.75), lower-casing and apostrophe normalisation |
| `char` | TF-IDF over character 3–5-grams within word boundaries |
| `mminilm` | paraphrase-multilingual-MiniLM-L12-v2, cosine similarity |
| `bge-m3` | BGE-M3 dense vectors (Ollama), cosine similarity |
| `rrf` | reciprocal rank fusion of `bm25` and `bge-m3`, k = 60 |
| `*_exp` | the query followed by a hypothetical answer from Qwen2.5-7B-Instruct (`expansion.py`) |

Measures: nDCG@10, Recall@1/10/100, MRR@10, the overlap of BM25 and BGE-M3 successes, index time and single-query
latency; paired bootstrap over questions and McNemar tests (`report.py`). `exploratory.py` holds the analyses added
after the planned runs (tuned fusion weights, retrieval by the hypothetical answer alone); see `CHANGELOG.md`.

## Data and models

- [WebFAQ retrieval](https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval), configurations `ukr-corpus`,
  `ukr-queries`, `ukr-qrels` (CC BY 4.0). `uafaq data` rebuilds the de-duplicated corpus (`data/corpus.jsonl`, not
  stored) and writes the question samples and collection pools stored in `data/`.
- Ollama 0.34.3 on an Apple M1 Pro: `bge-m3` (digest 790764642607) and `qwen2.5:7b-instruct` (845dbda0ea48, Q4_K_M);
  sentence-transformers for mMiniLM.

## Layout

```
src/uafaq/      data, text normalisation, retrievers, expansion, experiment, exploratory, latency, report
data/           queries_test.jsonl, queries_dev.jsonl, pools.json
results/        expansions.jsonl (generated answers with times), per_query.jsonl, summary.json, exploratory.json,
                timings.jsonl, latency.json, growth.png
tests/          unit tests
PROTOCOL.md     plan written before the runs; protocol_freeze.sha256 holds the hashes at that time
CHANGELOG.md    changes after the plan
```

## Usage

Python 3.11+ and [uv](https://docs.astral.sh/uv/). `requirements.lock.txt` lists the exact package versions used.

```bash
uv venv && uv pip install -r requirements.lock.txt && uv pip install -e ".[dev]" --no-deps
uv run --no-sync pytest
uv run --no-sync uafaq data            # corpus and samples from WebFAQ
uv run --no-sync uafaq run             # needs Ollama with bge-m3 and qwen2.5:7b-instruct; expansions resume from results/
uv run --no-sync uafaq exploratory
uv run --no-sync uafaq latency         # run on an otherwise idle machine
uv run --no-sync uafaq report
```

Cached vectors in `.cache/` are checked against the exact list of texts but not against the encoder, so check the model
digests above with `ollama list` before reusing them, or delete `.cache/` after a model update. `latency` does not time
BM25 over the expanded query.

## License

Code: MIT. The stored samples contain question ids and texts from WebFAQ (CC BY 4.0); see `THIRD_PARTY_NOTICES.md`.
