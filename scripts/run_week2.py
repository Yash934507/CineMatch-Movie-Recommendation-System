"""Run Week 2: python scripts/run_week2.py"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinematch.data import load_movielens  # noqa: E402
from cinematch.memory_cf import (  # noqa: E402
    MemoryBasedCF,
    deterministic_user_split,
    evaluate_predictions,
)


def main() -> None:
    started = time.perf_counter()
    ratings, movies, _ = load_movielens(ROOT / "data" / "raw")
    train, test = deterministic_user_split(ratings, test_fraction=0.2, seed=42)
    model = MemoryBasedCF(k_neighbors=30, min_common=3).fit(train)

    evaluation_sample = test.sample(n=min(2000, len(test)), random_state=42).reset_index(drop=True)
    user_predictions = [
        model.predict_user_based(row.userId, row.movieId)
        for row in evaluation_sample.itertuples(index=False)
    ]
    item_predictions = [
        model.predict_item_based(row.userId, row.movieId)
        for row in evaluation_sample.itertuples(index=False)
    ]
    actual = evaluation_sample["rating"].to_numpy()

    # Evaluation must use train-only data. The final demo may use every known
    # rating so it never recommends a title already present in the user's history.
    demo_model = MemoryBasedCF(k_neighbors=30, min_common=3).fit(ratings)
    star_wars_id = 260
    similar_star_wars = demo_model.similar_items(star_wars_id, movies, n=10)
    user_id = 1
    user_recommendations = demo_model.recommend(user_id, movies, method="user", n=5)
    item_recommendations = demo_model.recommend(user_id, movies, method="item", n=5)
    known_movie_ids = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    recommended_movie_ids = set(user_recommendations["movieId"]) | set(item_recommendations["movieId"])
    if known_movie_ids & recommended_movie_ids:
        raise AssertionError("Demo recommendations contain movies already rated by User 1")

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    similar_star_wars.to_csv(processed / "week2_similar_to_star_wars.csv", index=False)
    user_recommendations.to_csv(processed / "week2_user_user_recommendations_user_1.csv", index=False)
    item_recommendations.to_csv(processed / "week2_item_item_recommendations_user_1.csv", index=False)

    summary = {
        "split": {
            "strategy": "deterministic random holdout within each user",
            "seed": 42,
            "test_fraction": 0.2,
            "training_ratings": int(len(train)),
            "held_out_ratings": int(len(test)),
            "evaluation_sample_size": int(len(evaluation_sample)),
        },
        "model": {"neighbors": 30, "minimum_common_ratings": 3},
        "user_user": evaluate_predictions(actual, user_predictions),
        "item_item": evaluate_predictions(actual, item_predictions),
        "demo_user_id": user_id,
        "demo_trained_on_all_ratings": True,
        "demo_recommendations_exclude_known_ratings": True,
        "star_wars_movie_id": star_wars_id,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (processed / "week2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nSimilar to Star Wars: Episode IV - A New Hope (1977):")
    print(similar_star_wars.to_string(index=False))
    print("\nUser-User recommendations for User 1:")
    print(user_recommendations.to_string(index=False))
    print("\nItem-Item recommendations for User 1:")
    print(item_recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
