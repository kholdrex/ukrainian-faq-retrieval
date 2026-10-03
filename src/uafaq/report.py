"""Summary tables and the corpus-growth figure from results/per_query.jsonl."""
import json
from collections import defaultdict

import matplotlib
import numpy as np
from scipy.stats import binomtest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import BOOTSTRAP, POOL_SIZES, RESULTS, SEED
from .data import read_jsonl
from .metrics import bootstrap_mean_diff

SYSTEMS = ("bm25", "char", "mminilm", "bge-m3", "rrf", "bm25_exp", "bge-m3_exp", "rrf_exp")
POOLS = (*map(str, POOL_SIZES), "full")
METRICS = ("ndcg10", "r1", "r10", "mrr10", "r100")


def _mcnemar(a, b):
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return 1.0 if only_a + only_b == 0 else binomtest(only_a, only_a + only_b, 0.5).pvalue


def report() -> dict:
    rows = [r for r in read_jsonl(RESULTS / "per_query.jsonl") if r["split"] == "test"]
    table = defaultdict(dict)
    for r in rows:
        table[(r["pool"], r["system"])][r["query"]] = r
    queries = sorted(table[("full", "bm25")])
    vec = lambda pool, system, metric: np.array([table[(pool, system)][q][metric] for q in queries])

    out = {"n_queries": len(queries), "means": {}, "contrasts": {}, "overlap": {}}
    for pool in POOLS:
        for system in SYSTEMS:
            out["means"][f"{pool}|{system}"] = {m: float(vec(pool, system, m).mean()) for m in METRICS}

    single = max(("bm25", "char", "mminilm", "bge-m3"), key=lambda s: out["means"][f"full|{s}"]["ndcg10"])
    for a, b in ((single, "rrf"), ("rrf", "rrf_exp"), ("bge-m3", "bge-m3_exp"), ("bm25", "bm25_exp")):
        for pool in POOLS:
            diff, low, high = bootstrap_mean_diff(vec(pool, b, "ndcg10"), vec(pool, a, "ndcg10"), BOOTSTRAP, SEED)
            p = _mcnemar(vec(pool, b, "r10").astype(bool), vec(pool, a, "r10").astype(bool))
            out["contrasts"][f"{pool}|{b}-{a}"] = {"d_ndcg10": diff, "ci": [low, high], "p_r10": p}

    for pool in POOLS:
        lexical, dense = vec(pool, "bm25", "r10").astype(bool), vec(pool, "bge-m3", "r10").astype(bool)
        fused = vec(pool, "rrf", "r10").astype(bool)
        out["overlap"][pool] = {"both": int((lexical & dense).sum()), "lexical_only": int((lexical & ~dense).sum()),
                                "dense_only": int((~lexical & dense).sum()), "neither": int((~lexical & ~dense).sum()),
                                "rrf_r10": int(fused.sum()), "oracle_union": int((lexical | dense).sum())}
    out["best_single"] = single
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    _figure(out)
    return out


def _figure(out):
    sizes = [*POOL_SIZES, 67721]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    styles = {"bm25": ("o-", "BM25"), "char": ("s-", "char 3-5-gram TF-IDF"), "mminilm": ("^-", "mMiniLM"),
              "bge-m3": ("D-", "BGE-M3"), "rrf": ("v--", "RRF (BM25 + BGE-M3)"),
              "rrf_exp": ("x:", "RRF with LLM expansion")}
    for system, (style, label) in styles.items():
        ax.plot(sizes, [out["means"][f"{p}|{system}"]["ndcg10"] for p in POOLS], style, color="black",
                markersize=4, linewidth=1, label=label, markerfacecolor="white")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Answers in the collection", fontsize=9)
    ax.set_ylabel("nDCG@10", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "growth.png", dpi=600)
    fig.savefig(RESULTS / "growth.svg")
    plt.close(fig)
