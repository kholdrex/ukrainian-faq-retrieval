import numpy as np

from uafaq.metrics import bootstrap_mean_diff, first_hit, per_query
from uafaq.retrieval import BM25, CharTfidf, rrf, top_k
from uafaq.text import normalise, words

DOCS = ["Доставка по Україні триває два дні.", "Оплатити замовлення можна карткою.",
        "Готель має безкоштовну парковку.", "Повернення товару можливе протягом 14 днів."]


def test_normalise_apostrophes():
    assert normalise("Пам’ять і пам'ять") == "пам'ять і пам'ять"
    assert words("Ім’я, прізвище!") == ["ім'я", "прізвище"]


def test_top_k_orders_by_score():
    scores = np.array([[0.1, 0.9, 0.5, 0.3]])
    assert top_k(scores, 3).tolist() == [[1, 2, 3]]


def test_bm25_and_char_find_matching_document():
    for retriever in (BM25().fit(DOCS), CharTfidf().fit(DOCS * 2)):
        assert retriever.search(["Як оплатити карткою?"])[0][0] % 4 == 1


def test_rrf_rewards_agreement():
    fused = rrf(np.array([[0, 1, 2]]), np.array([[1, 0, 2]]), depth=3)
    assert set(fused[0][:2]) == {0, 1} and fused[0][2] == 2


def test_per_query_measures():
    m = per_query(["a", "b", "c"], {"b"})
    assert m["rank"] == 2 and m["r1"] == 0 and m["r10"] == 1
    assert abs(m["ndcg10"] - 1 / np.log2(3)) < 1e-9 and m["mrr10"] == 0.5
    assert first_hit(["a"], {"z"}) is None


def test_bootstrap_difference():
    diff, low, high = bootstrap_mean_diff(np.ones(50), np.zeros(50), 200, 0)
    assert diff == 1 and low == high == 1
