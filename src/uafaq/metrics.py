import numpy as np


def first_hit(ranking: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant document, or None."""
    for rank, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return rank
    return None


def per_query(ranking: list[str], relevant: set[str]) -> dict:
    rank = first_hit(ranking, relevant)
    gains = [1.0 / np.log2(r + 1) for r, doc in enumerate(ranking[:10], start=1) if doc in relevant]
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), 10) + 1))
    return {
        "rank": rank,
        "ndcg10": sum(gains) / ideal,
        "mrr10": 1.0 / rank if rank and rank <= 10 else 0.0,
        "r1": float(rank == 1),
        "r10": float(bool(rank) and rank <= 10),
        "r100": float(bool(rank) and rank <= 100),
    }


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
