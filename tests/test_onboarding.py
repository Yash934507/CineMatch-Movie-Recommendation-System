"""Onboarding math, temporal isolation, candidate parity, and API compatibility."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cinematch.evaluation import ranking_metrics
from cinematch.onboarding import (
    aggregate_details, chronological_profile, evaluate_onboarding, fold_in,
    generate_candidates, partition_history, rank_fold_in, score_fold_in,
    select_product_policy,
)


class TinyTrainset:
    global_mean = 3.0

    def to_inner_iid(self, movie_id):
        if movie_id not in range(1, 7):
            raise ValueError("Unknown movie")
        return movie_id - 1


@pytest.fixture
def model():
    return SimpleNamespace(trainset=TinyTrainset(), bi=np.zeros(6),
                           qi=np.array([[1.0], [-1.0], [0.0], [1.0], [-1.0], [0.5]]))


@pytest.fixture
def history():
    return pd.DataFrame({"userId": [1] * 6, "movieId": [6, 2, 4, 1, 5, 3],
                         "timestamp": [60, 20, 40, 10, 50, 30], "rating": [4, 2, 5, 4, 1, 3]})


def test_chronological_order_and_strict_future(history):
    (early, later), reason = chronological_profile(history, 3, set(range(1, 7)))
    assert reason is None
    assert early.movieId.tolist() == [1, 2, 3]
    assert later.movieId.tolist() == [4, 5, 6]
    assert early.timestamp.max() < later.timestamp.min()


def test_future_values_cannot_change_profile(history, model):
    (early, _), _ = chronological_profile(history, 3, set(range(1, 7)))
    altered = history.copy()
    altered.loc[altered.timestamp > 30, "rating"] = 5.0
    (again, _), _ = chronological_profile(altered, 3, set(range(1, 7)))
    pd.testing.assert_frame_equal(early, again)
    a = fold_in(model, list(early[["movieId", "rating"]].itertuples(index=False, name=None)))
    b = fold_in(model, list(again[["movieId", "rating"]].itertuples(index=False, name=None)))
    np.testing.assert_array_equal(a.factors, b.factors)


def test_timestamp_ties_not_treated_as_future(history):
    history.loc[history.movieId.isin([3, 4]), "timestamp"] = 30
    (early, later), _ = chronological_profile(history, 3, set(range(1, 7)))
    assert early.movieId.tolist() == [1, 2, 3]
    assert later.movieId.tolist() == [5, 6]


@pytest.mark.parametrize("case,reason", [
    ("short", "insufficient_history"), ("unknown", "unknown_onboarding_item"),
    ("irrelevant", "no_later_relevant_movie"),
])
def test_eligibility(history, case, reason):
    known = set(range(1, 7))
    if case == "short":
        history = history.head(3)
    elif case == "unknown":
        known.remove(1)
    else:
        history.loc[history.timestamp > 30, "rating"] = 3.0
    assert chronological_profile(history, 3, known) == (None, reason)


def test_duplicate_history_rejected(history):
    with pytest.raises(ValueError, match="Duplicate"):
        chronological_profile(pd.concat([history, history.head(1)]), 3, set(range(1, 7)))


@pytest.mark.parametrize("ratings,message", [
    ([(1, 4), (1, 3), (2, 4)], "only be rated once"),
    ([(1, 4), (2, 3)], "At least 3"),
    ([(1, np.nan), (2, 3), (3, 4)], "finite"),
    ([(1, 6), (2, 3), (3, 4)], "between"),
    ([(1, 4), (2, 3), (999, 4)], "no learned"),
])
def test_fold_in_validation(model, ratings, message):
    with pytest.raises(ValueError, match=message):
        fold_in(model, ratings)


def test_rating_profiles_change_ranking_and_preserve_model(model):
    original = model.qi.copy()
    a = fold_in(model, [(1, 5), (2, 1), (3, 3)])
    b = fold_in(model, [(1, 1), (2, 5), (3, 3)])
    ranked_a = rank_fold_in(model, a, list(range(1, 7)))
    ranked_b = rank_fold_in(model, b, list(range(1, 7)))
    assert ranked_a[0][0] == 4
    assert ranked_b[0][0] == 5
    assert not {1, 2, 3} & {movie for movie, _ in ranked_a}
    np.testing.assert_array_equal(original, model.qi)
    assert all(0.5 <= score <= 5 for _, score in ranked_a + ranked_b)


def test_unknown_item_scoring(model):
    profile = fold_in(model, [(1, 4), (2, 3), (3, 3)])
    assert score_fold_in(model, profile, [999])[0] == pytest.approx(np.clip(3 + profile.bias, 0.5, 5))


def test_fold_in_solves_known_normal_equation(model):
    profile = fold_in(model, [(1, 5), (2, 1), (3, 3)])
    assert profile.bias == pytest.approx(0)
    assert profile.factors[0] == pytest.approx(4 / 2.1)


def test_candidates_include_all_later_and_no_observed_negatives(history):
    later = history.loc[history.timestamp > 30]
    a, n = generate_candidates(history, later, set(range(1, 20)), negative_samples=4, user_id=1)
    b, _ = generate_candidates(history, later, set(range(1, 20)), negative_samples=4, user_id=1)
    assert a == b and n == 4
    assert set(later.movieId).issubset(a)
    assert (set(a) - set(later.movieId)).isdisjoint(history.movieId)


def test_negative_pool_shortfall(history):
    candidates, n = generate_candidates(history, history.iloc[:1], {1, 2, 7}, negative_samples=100)
    assert n == 1
    assert set(candidates) == {6, 7}


def test_known_ranking_metrics():
    result = ranking_metrics([1, 2, 3, 4], {2, 4}, k=10)
    assert result["precision_at_k"] == pytest.approx(0.2)
    assert result["recall_at_k"] == 1
    assert result["hit_rate_at_k"] == 1
    expected = (1 / np.log2(3) + 1 / np.log2(5)) / (1 + 1 / np.log2(3))
    assert result["ndcg_at_k"] == pytest.approx(expected)
    assert ranking_metrics([1], {2}, k=10)["ndcg_at_k"] == 0


def small_dataset():
    rows = [(uid, movie, float(1 + (uid + movie) % 5), movie)
            for uid in range(1, 5) for movie in range(1, 21)]
    rows += [(uid, movie, 4.0 if movie % 2 else 3.0, 100 + movie)
             for uid in range(10, 18) for movie in range(1, 16)]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    movies = pd.DataFrame({"movieId": range(1, 21), "title": [str(x) for x in range(1, 21)], "genres": "Drama"})
    return ratings, movies


def test_partition_has_no_user_or_temporal_leakage():
    ratings, _ = small_dataset()
    train, cohorts = partition_history(ratings, 100, 42)
    assert train.timestamp.max() < 100
    evaluated = set(cohorts["validation"] + cohorts["test"])
    assert evaluated.isdisjoint(train.userId)
    assert set(cohorts["validation"]).isdisjoint(cohorts["test"])
    altered = ratings.copy()
    altered.loc[altered.timestamp >= 100, "rating"] = 0.5
    again, other = partition_history(altered, 100, 42)
    pd.testing.assert_frame_equal(train, again)
    assert cohorts == other


def test_orchestration_is_deterministic_and_candidates_identical():
    ratings, movies = small_dataset()
    metrics, details, summary = evaluate_onboarding(ratings, movies, cutoff=100, negative_samples=4)
    again, repeated, other = evaluate_onboarding(ratings.sample(frac=1, random_state=7), movies, cutoff=100, negative_samples=4)
    pd.testing.assert_frame_equal(metrics, again)
    pd.testing.assert_frame_equal(details, repeated)
    assert summary == other
    for _, group in details.groupby(["cohort", "onboarding_size", "userId"]):
        assert group.candidate_movie_ids.nunique() == 1
        assert group.candidate_sha256.nunique() == 1
        row = group.iloc[0]
        assert set(json.loads(row.onboarding_movie_ids)).isdisjoint(json.loads(row.candidate_movie_ids))
        assert row.onboarding_end_timestamp < row.test_start_timestamp
    row = metrics.iloc[0]
    selected = details.loc[(details.cohort == row.cohort) & (details.onboarding_size == row.onboarding_size) & (details.strategy == row.strategy)]
    union = set().union(*(set(json.loads(x)) for x in selected.recommended_movie_ids))
    assert row.catalog_coverage == len(union) / len(movies)
    with pytest.raises(ValueError, match="validation users only"):
        select_product_policy(details)


def test_evaluation_training_and_selection_call_order(monkeypatch):
    import cinematch.onboarding as module
    ratings, movies = small_dataset()
    real_fit, real_select = module.fit_svd, module.select_product_policy
    calls = []

    def fit(train, **kwargs):
        assert (train.timestamp < 100).all()
        assert set(train.userId) == {1, 2, 3, 4}
        calls.append("fit")
        return real_fit(train, **kwargs)

    def select(validation):
        assert set(validation.cohort) == {"validation"}
        calls.append("select")
        return real_select(validation)

    monkeypatch.setattr(module, "fit_svd", fit)
    monkeypatch.setattr(module, "select_product_policy", select)
    module.evaluate_onboarding(ratings, movies, cutoff=100, negative_samples=4)
    assert calls == ["fit", "select"]
