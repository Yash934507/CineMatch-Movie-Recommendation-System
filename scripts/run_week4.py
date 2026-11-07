"""Run Week 4 final evaluation: python scripts/run_week4.py"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / "work" / "surprise_data"))
sys.path.insert(0, str(ROOT / "src"))

from cinematch.data import load_movielens  # noqa: E402
from cinematch.evaluation import (  # noqa: E402
    build_ranking_candidates,
    evaluate_ranking,
    popularity_fallback,
)
from cinematch.memory_cf import (  # noqa: E402
    MemoryBasedCF,
    deterministic_user_split,
    evaluate_predictions,
)
from cinematch.svd_model import (  # noqa: E402
    fit_svd,
    predict_frame,
    recommend_svd,
    regression_metrics,
)


def save_charts(rating_results: pd.DataFrame, ranking_results: pd.DataFrame) -> None:
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rating_results.set_index("model")[["rmse", "mae"]].plot.bar(
        ax=ax, color=["#7c3aed", "#0f766e"]
    )
    ax.set(title="Rating prediction error on the shared test sample", ylabel="Error (lower is better)", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(figures / "week4_rating_error_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ranking_results.set_index("model")[["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]].plot.bar(ax=ax)
    ax.set(title="Top-10 ranking quality on sampled candidates", ylabel="Mean score (higher is better)", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric", ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "week4_ranking_comparison.png", dpi=160)
    plt.close(fig)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rendered = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, separator, *rows])


def write_report(summary: dict, rating_results: pd.DataFrame, ranking_results: pd.DataFrame) -> None:
    rating_table = frame_to_markdown(rating_results)
    ranking_table = frame_to_markdown(ranking_results)
    text = f"""# CineMatch Final Evaluation Report

## Evaluation design

- MovieLens Small: 100,836 explicit ratings.
- Shared per-user 80/20 holdout with seed 42.
- Rating evaluation: identical deterministic sample of 2,000 held-out ratings.
- Ranking evaluation: {summary['ranking_protocol']['evaluated_users']} users with at least one held-out rating >= {summary['ranking_protocol']['relevance_threshold']}.
- Each ranking candidate set contains all of that user's held-out movies plus {summary['ranking_protocol']['sampled_unseen_candidates']} sampled unseen movies.
- K = {summary['ranking_protocol']['k']}.

Unrated sampled movies are evaluation candidates, not confirmed dislikes. Ranking results must therefore be interpreted under this sampled-candidate protocol.

## Rating prediction

{rating_table}

Best rating model: **{summary['selection']['best_rating_model']}**.

## Top-K ranking

{ranking_table}

Best ranking model by Precision@10, with NDCG@10 and RMSE as tie-breakers: **{summary['selection']['best_ranking_model']}**.

## Final selection

The recommended model for the CineMatch demo is **{summary['selection']['recommended_model']}** because the product's primary goal is ranked discovery. This selection is based on measured Precision@10 first, not on an assumption that matrix factorization must win.

## Cold start

For a new user with no ratings, collaborative filtering has no personal signal. CineMatch therefore uses a Bayesian-weighted popularity list with a prior weight of 50 ratings. This reduces the chance that a movie with one unusually high rating outranks broadly liked movies. A future hybrid can ask new users to select genres or rate several onboarding titles.

## Limitations

- Precision and recall are measured with sampled negatives because MovieLens does not identify true dislikes among unrated movies.
- The data is a historical development snapshot and contains no demographics or current catalog availability.
- Offline accuracy does not measure novelty, diversity, fairness, or real user satisfaction.
- Latent factors are not automatically interpretable genre labels.
"""
    (ROOT / "reports" / "week4_evaluation_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    ratings, movies, _ = load_movielens(ROOT / "data" / "raw")
    train, test = deterministic_user_split(ratings, test_fraction=0.2, seed=42)
    sample = test.sample(n=min(2000, len(test)), random_state=42).reset_index(drop=True)

    memory = MemoryBasedCF(k_neighbors=30, min_common=3).fit(train)
    svd = fit_svd(train, n_factors=20, n_epochs=20, random_state=42)

    user_predictions = [memory.predict_user_based(row.userId, row.movieId) for row in sample.itertuples(index=False)]
    item_predictions = [memory.predict_item_based(row.userId, row.movieId) for row in sample.itertuples(index=False)]
    svd_predictions = predict_frame(svd, sample)
    actual = sample["rating"].to_numpy()
    user_metrics = evaluate_predictions(actual, user_predictions)
    item_metrics = evaluate_predictions(actual, item_predictions)
    svd_metrics = regression_metrics(svd_predictions)
    rating_results = pd.DataFrame(
        [
            {"model": "User-User", "rmse": user_metrics["rmse"], "mae": user_metrics["mae"]},
            {"model": "Item-Item", "rmse": item_metrics["rmse"], "mae": item_metrics["mae"]},
            {"model": "SVD", "rmse": svd_metrics["rmse"], "mae": svd_metrics["mae"]},
        ]
    )

    candidates = build_ranking_candidates(
        train,
        test,
        movies["movieId"].to_numpy(),
        user_limit=100,
        negative_samples=100,
        relevance_threshold=4.0,
        seed=42,
    )
    predictors = {
        "User-User": lambda user_id, movie_id: memory.predict_user_based(user_id, movie_id).value,
        "Item-Item": lambda user_id, movie_id: memory.predict_item_based(user_id, movie_id).value,
        "SVD": lambda user_id, movie_id: float(svd.predict(user_id, movie_id).est),
    }
    ranking_results, ranking_details = evaluate_ranking(
        predictors, candidates, k=10, catalog_size=len(movies)
    )

    best_rating = rating_results.sort_values(["rmse", "mae", "model"]).iloc[0]["model"]
    selection_frame = ranking_results.merge(rating_results[["model", "rmse"]], on="model")
    best_ranking = selection_frame.sort_values(
        ["precision_at_k", "ndcg_at_k", "rmse", "model"],
        ascending=[False, False, True, True],
    ).iloc[0]["model"]
    recommended = str(best_ranking)

    cold_start = popularity_fallback(ratings, movies, n=10, prior_weight=50)
    if recommended == "SVD":
        final_model = fit_svd(ratings, n_factors=20, n_epochs=20, random_state=42)
        final_recommendations = recommend_svd(final_model, 1, ratings, movies, n=10)
    else:
        final_memory = MemoryBasedCF(k_neighbors=30, min_common=3).fit(ratings)
        method = "user" if recommended == "User-User" else "item"
        final_recommendations = final_memory.recommend(1, movies, method=method, n=10)
    known_user_1 = set(ratings.loc[ratings["userId"] == 1, "movieId"])
    if known_user_1 & set(final_recommendations["movieId"]):
        raise AssertionError("Final recommendations contain movies already rated by User 1")

    processed = ROOT / "data" / "processed"
    rating_results.to_csv(processed / "week4_rating_comparison.csv", index=False)
    ranking_results.to_csv(processed / "week4_ranking_comparison.csv", index=False)
    ranking_details.to_csv(processed / "week4_ranking_user_details.csv", index=False)
    cold_start.to_csv(processed / "week4_cold_start_recommendations.csv", index=False)
    final_recommendations.to_csv(processed / "week4_final_recommendations_user_1.csv", index=False)

    summary = {
        "rating_protocol": {
            "split": "deterministic per-user 80/20 holdout",
            "seed": 42,
            "training_ratings": int(len(train)),
            "held_out_ratings": int(len(test)),
            "evaluation_sample_size": int(len(sample)),
        },
        "ranking_protocol": {
            "evaluated_users": len(candidates),
            "user_selection_seed": 42,
            "relevance_threshold": 4.0,
            "sampled_unseen_candidates": 100,
            "k": 10,
        },
        "rating_results": rating_results.set_index("model").to_dict(orient="index"),
        "ranking_results": ranking_results.set_index("model").to_dict(orient="index"),
        "selection": {
            "best_rating_model": str(best_rating),
            "best_ranking_model": str(best_ranking),
            "recommended_model": recommended,
            "primary_selection_metric": "precision_at_10",
            "tie_breakers": ["ndcg_at_10", "rmse"],
        },
        "cold_start": {"strategy": "Bayesian-weighted popularity", "prior_weight": 50},
        "final_demo": {
            "user_id": 1,
            "model": recommended,
            "recommendation_count": int(len(final_recommendations)),
            "excludes_known_ratings": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (processed / "week4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_charts(rating_results, ranking_results)
    write_report(summary, rating_results, ranking_results)

    print(json.dumps(summary, indent=2))
    print("\nCold-start recommendations:")
    print(cold_start.to_string(index=False))
    print("\nFinal Top-10 recommendations for User 1:")
    print(final_recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
