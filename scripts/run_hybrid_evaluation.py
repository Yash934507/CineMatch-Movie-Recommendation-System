"""Tune and evaluate hybrid diversity/novelty reranking with validation data.

The script preserves the selected hybrid relevance model, builds a shared candidate
pool for every ranking variant, chooses lambda only on validation data, and then
runs exactly one final test evaluation with that selected value.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / "work" / "surprise_data"))
sys.path.insert(0, str(ROOT / "src"))

from cinematch.data import load_movielens  # noqa: E402
from cinematch.evaluation import build_ranking_candidates, ranking_metrics  # noqa: E402
from cinematch.hybrid import (  # noqa: E402
    build_genre_profile,
    candidate_genre_similarity,
    compute_item_novelty,
    genre_coverage,
    intra_list_diversity,
    mean_topk_novelty,
    mmr_rerank,
    novelty_rerank,
    rank_hybrid_candidates,
    unique_genre_count,
)
from cinematch.memory_cf import deterministic_user_split  # noqa: E402
from cinematch.svd_model import fit_svd  # noqa: E402


BASELINE_WEIGHTS = {"collaborative": 0.7, "genre": 0.3, "popularity": 0.0}
VALIDATION_LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def hybrid_scores_for_user(
    model,
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    candidate_ids: list[int],
    *,
    weights: dict[str, float],
) -> dict[int, float]:
    profile = build_genre_profile(train, movies, int(user_id))
    collaborative = {int(movie_id): float(model.predict(int(user_id), int(movie_id)).est) for movie_id in candidate_ids}
    genre = {
        int(movie_id): float(candidate_genre_similarity(profile, movies.loc[movies["movieId"] == int(movie_id), "genres"].iloc[0]))
        for movie_id in candidate_ids
    }
    ranked = rank_hybrid_candidates(collaborative, genre, weights=weights)
    return {movie_id: score for movie_id, score in ranked}


def evaluate_strategy(
    strategy_name: str,
    order_by_user: dict[int, list[int]],
    candidates_by_user: dict[int, dict[str, object]],
    movies: pd.DataFrame,
    novelty_by_movie: dict[int, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    detail_rows = []
    aggregate = []
    recommended_catalog: set[int] = set()
    for user_id in sorted(order_by_user):
        ranked = order_by_user[user_id]
        relevant = set(int(item) for item in candidates_by_user[user_id]["relevant"])
        metrics = ranking_metrics(ranked, relevant, k=10)
        recommended_catalog.update(ranked[:10])
        detail_rows.append(
            {
                "strategy": strategy_name,
                "userId": int(user_id),
                "candidate_items": len(candidates_by_user[user_id]["candidates"]),
                "relevant_items": len(relevant),
                "recommended_movie_ids": json.dumps(ranked[:10]),
                "precision_at_k": metrics["precision_at_k"],
                "recall_at_k": metrics["recall_at_k"],
                "hit_rate_at_k": metrics["hit_rate_at_k"],
                "ndcg_at_k": metrics["ndcg_at_k"],
                "intra_list_diversity": intra_list_diversity(ranked[:10], movies),
                "mean_novelty": mean_topk_novelty(ranked[:10], novelty_by_movie),
                "genre_count": unique_genre_count(ranked[:10], movies),
            }
        )
    frame = pd.DataFrame(detail_rows)
    aggregate = {
        "strategy": strategy_name,
        "evaluated_users": int(len(frame)),
        "precision_at_k": float(frame["precision_at_k"].mean()),
        "recall_at_k": float(frame["recall_at_k"].mean()),
        "hit_rate_at_k": float(frame["hit_rate_at_k"].mean()),
        "ndcg_at_k": float(frame["ndcg_at_k"].mean()),
        "intra_list_diversity": float(frame["intra_list_diversity"].mean()),
        "mean_novelty": float(frame["mean_novelty"].mean()),
        "genre_count": float(frame["genre_count"].mean()),
        "catalog_coverage": float(len(recommended_catalog) / len(movies)),
    }
    return pd.DataFrame(detail_rows), aggregate


def select_lambda(validation_results: list[dict[str, float]], baseline_ndcg: float) -> tuple[float, str]:
    """Select the best lambda under the explicit NDCG-first relevance constraint."""
    candidates = [row for row in validation_results if row["candidate_ndcg"] >= baseline_ndcg * 0.98]
    if not candidates:
        return 1.0, "baseline"
    chosen = max(candidates, key=lambda row: (row["intra_list_diversity"], row["novelty"], row["lambda"]))
    return float(chosen["lambda"]), "selected"


def score_baseline(
    model,
    train: pd.DataFrame,
    movies: pd.DataFrame,
    candidates_by_user: dict[int, dict[str, object]],
    weights: dict[str, float],
) -> dict[int, list[int]]:
    output: dict[int, list[int]] = {}
    for user_id, payload in candidates_by_user.items():
        candidate_ids = [int(item) for item in payload["candidates"]]
        hybrid_map = hybrid_scores_for_user(model, train, movies, int(user_id), candidate_ids, weights=weights)
        output[int(user_id)] = [movie_id for movie_id, _ in sorted(hybrid_map.items(), key=lambda item: (-item[1], item[0]))[:50]]
    return output


def main() -> None:
    ratings, movies, _ = load_movielens(ROOT / "data" / "raw")
    train, validation = deterministic_user_split(ratings, test_fraction=0.2, seed=42)
    train_for_final, final_test = deterministic_user_split(train, test_fraction=0.2, seed=43)
    model = fit_svd(train, n_factors=20, n_epochs=20, random_state=42)
    candidates = build_ranking_candidates(
        train,
        validation,
        movies["movieId"].to_numpy(),
        user_limit=100,
        negative_samples=100,
        relevance_threshold=4.0,
        seed=42,
    )
    novelty_by_movie = compute_item_novelty(train, movies)
    baseline_order = score_baseline(model, train, movies, candidates, BASELINE_WEIGHTS)
    baseline_details, baseline_metrics = evaluate_strategy("baseline", baseline_order, candidates, movies, novelty_by_movie)
    baseline_ndcg = float(baseline_metrics["ndcg_at_k"])

    validation_table = []
    for lambda_value in VALIDATION_LAMBDAS:
        reranked_orders: dict[int, list[int]] = {}
        for user_id, payload in candidates.items():
            candidate_ids = [int(item) for item in payload["candidates"]]
            hybrid_map = hybrid_scores_for_user(model, train, movies, int(user_id), candidate_ids, weights=BASELINE_WEIGHTS)
            top_50 = [movie_id for movie_id, _ in sorted(hybrid_map.items(), key=lambda item: (-item[1], item[0]))[:50]]
            relevant = set(int(item) for item in payload["relevant"])
            if lambda_value == 1.0:
                reranked = top_50
            else:
                reranked = mmr_rerank(hybrid_map, top_50, movies, lambda_=float(lambda_value), k=10)
            if not reranked:
                reranked = top_50[:10]
            reranked_orders[int(user_id)] = reranked[:10]
        _, aggregate = evaluate_strategy(f"mmr-lambda-{lambda_value}", reranked_orders, candidates, movies, novelty_by_movie)
        validation_table.append(
            {
                "lambda": lambda_value,
                "candidate_ndcg": float(aggregate["ndcg_at_k"]),
                "precision": float(aggregate["precision_at_k"]),
                "recall": float(aggregate["recall_at_k"]),
                "hit_rate": float(aggregate["hit_rate_at_k"]),
                "intra_list_diversity": float(aggregate["intra_list_diversity"]),
                "novelty": float(aggregate["mean_novelty"]),
                "genre_coverage": float(aggregate["genre_count"]),
                "catalog_coverage": float(aggregate["catalog_coverage"]),
            }
        )

    selected_lambda, selection_mode = select_lambda(validation_table, baseline_ndcg)
    selected = next((row for row in validation_table if row["lambda"] == selected_lambda), None)
    final_candidates = build_ranking_candidates(
        train_for_final,
        final_test,
        movies["movieId"].to_numpy(),
        user_limit=100,
        negative_samples=100,
        relevance_threshold=4.0,
        seed=42,
    )
    final_order = {}
    for user_id, payload in final_candidates.items():
        candidate_ids = [int(item) for item in payload["candidates"]]
        hybrid_map = hybrid_scores_for_user(model, train_for_final, movies, int(user_id), candidate_ids, weights=BASELINE_WEIGHTS)
        top_50 = [movie_id for movie_id, _ in sorted(hybrid_map.items(), key=lambda item: (-item[1], item[0]))[:50]]
        if selected_lambda == 1.0:
            final_order[int(user_id)] = top_50[:10]
        else:
            final_order[int(user_id)] = mmr_rerank(hybrid_map, top_50, movies, lambda_=selected_lambda, k=10)[:10]
    _, final_metrics = evaluate_strategy("final-test-mmr", final_order, final_candidates, movies, novelty_by_movie)

    summary = {
        "selection_rule": "Validation NDCG@10 must remain within 2% relative of the baseline; among passing candidates choose highest intra-list diversity, tie-break by novelty then larger lambda.",
        "baseline": {
            "strategy": "hybrid-0.7-genre-0.3",
            "weights": BASELINE_WEIGHTS,
            "ndcg_at_k": baseline_metrics["ndcg_at_k"],
            "precision_at_k": baseline_metrics["precision_at_k"],
            "recall_at_k": baseline_metrics["recall_at_k"],
            "hit_rate_at_k": baseline_metrics["hit_rate_at_k"],
        },
        "validation_lambda_grid": validation_table,
        "selected_lambda": selected_lambda,
        "selection_mode": selection_mode,
        "final_test": {
            "strategy": "hybrid-0.7-genre-0.3-mmr",
            "lambda": selected_lambda,
            "ndcg_at_k": final_metrics["ndcg_at_k"],
            "precision_at_k": final_metrics["precision_at_k"],
            "recall_at_k": final_metrics["recall_at_k"],
            "hit_rate_at_k": final_metrics["hit_rate_at_k"],
            "intra_list_diversity": final_metrics["intra_list_diversity"],
            "mean_novelty": final_metrics["mean_novelty"],
            "genre_count": final_metrics["genre_count"],
            "catalog_coverage": final_metrics["catalog_coverage"],
        },
    }

    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "hybrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    details = []
    for name, orders in [("baseline", baseline_order), ("final-test-mmr", final_order)]:
        for user_id, recommendation_ids in orders.items():
            details.append({
                "strategy": name,
                "userId": int(user_id),
                "recommended_movie_ids": recommendation_ids,
            })
    pd.DataFrame(details).to_csv(processed_dir / "hybrid_user_details.csv", index=False)

    report = ROOT / "reports"
    report.mkdir(parents=True, exist_ok=True)
    report_path = report / "hybrid_recommendation_evaluation.md"
    report_path.write_text(
        "# Hybrid recommendation evaluation\n\n"
        "## Validation selection rule\n\n"
        "- Baseline is the selected hybrid model with weights 0.7 collaborative, 0.3 genre, 0.0 popularity.\n"
        "- A reranking configuration qualifies only if validation NDCG@10 remains within 2% relative of the baseline.\n"
        "- Among qualifying configurations, choose the highest intra-list diversity; tie-break by novelty and then by larger lambda.\n\n"
        f"- Selected lambda: {selected_lambda:.2f}\n"
        f"- Validation baseline NDCG@10: {baseline_ndcg:.4f}\n"
        f"- Final test NDCG@10: {final_metrics['ndcg_at_k']:.4f}\n\n"
        "## Validation results\n\n"
        + "\n".join(
            f"- lambda={row['lambda']:.2f}: NDCG={row['candidate_ndcg']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, HitRate={row['hit_rate']:.4f}, Diversity={row['intra_list_diversity']:.4f}, Novelty={row['novelty']:.4f}."
            for row in validation_table
        )
        + "\n\n## Final test summary\n\n"
        + "\n".join(
            f"- {key}: {value}"
            for key, value in final_metrics.items()
        ),
        encoding="utf-8",
    )

    print(json.dumps({"selected_lambda": selected_lambda, "selection_mode": selection_mode, "baseline_ndcg": baseline_ndcg, "final_test_ndcg": final_metrics["ndcg_at_k"]}, indent=2))


if __name__ == "__main__":
    main()
