"""Run Week 3: python scripts/run_week3.py"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / "work" / "surprise_data"))
sys.path.insert(0, str(ROOT / "src"))

import surprise  # noqa: E402

from cinematch.data import load_movielens  # noqa: E402
from cinematch.memory_cf import deterministic_user_split  # noqa: E402
from cinematch.svd_model import (  # noqa: E402
    fit_svd,
    latent_feature_extremes,
    predict_frame,
    recommend_svd,
    regression_metrics,
    save_model,
    tune_factors,
)


def main() -> None:
    started = time.perf_counter()
    ratings, movies, _ = load_movielens(ROOT / "data" / "raw")

    # Recreate Week 2's outer split. The test sample stays untouched while
    # factor count is selected on a validation subset of the training data.
    train_full, test = deterministic_user_split(ratings, test_fraction=0.2, seed=42)
    tune_train, validation = deterministic_user_split(train_full, test_fraction=0.125, seed=43)
    tuning, best_factors = tune_factors(
        tune_train,
        validation,
        factors=(20, 50, 100),
        n_epochs=20,
        random_state=42,
    )

    final_model = fit_svd(train_full, n_factors=best_factors, n_epochs=20, random_state=42)
    evaluation_sample = test.sample(n=min(2000, len(test)), random_state=42).reset_index(drop=True)
    test_predictions = predict_frame(final_model, evaluation_sample)
    test_metrics = regression_metrics(test_predictions)

    # Refit on all known ratings for user-facing recommendations.
    demo_model = fit_svd(ratings, n_factors=best_factors, n_epochs=20, random_state=42)
    recommendations = recommend_svd(demo_model, 1, ratings, movies, n=5)
    known = set(ratings.loc[ratings["userId"] == 1, "movieId"])
    if known & set(recommendations["movieId"]):
        raise AssertionError("SVD recommendations contain movies already rated by User 1")
    factors = latent_feature_extremes(demo_model, movies, ratings)

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(processed / "week3_svd_factor_tuning.csv", index=False)
    test_predictions.to_csv(processed / "week3_svd_test_predictions.csv", index=False)
    recommendations.to_csv(processed / "week3_svd_recommendations_user_1.csv", index=False)
    factors.to_csv(processed / "week3_latent_feature_extremes.csv", index=False)
    save_model(demo_model, ROOT / "models" / "svd_model.pkl")

    summary = {
        "library": {"name": "scikit-surprise", "version": surprise.__version__},
        "split": {
            "outer_strategy": "same deterministic per-user 80/20 split as Week 2",
            "outer_seed": 42,
            "training_ratings": int(len(train_full)),
            "held_out_ratings": int(len(test)),
            "evaluation_sample_size": int(len(evaluation_sample)),
            "validation_strategy": "12.5% of outer training data (10% of all ratings)",
            "validation_seed": 43,
            "tuning_train_ratings": int(len(tune_train)),
            "validation_ratings": int(len(validation)),
        },
        "tuned_n_factors": [20, 50, 100],
        "selected_n_factors": best_factors,
        "n_epochs": 20,
        "test": test_metrics,
        "demo_user_id": 1,
        "demo_trained_on_all_ratings": True,
        "demo_recommendations_exclude_known_ratings": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (processed / "week3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nFactor tuning (sorted by validation RMSE):")
    print(tuning.to_string(index=False))
    print("\nSVD recommendations for User 1:")
    print(recommendations.to_string(index=False))
    print("\nLatent-factor extremes (exploratory; factors are not automatically named):")
    print(factors.to_string(index=False))


if __name__ == "__main__":
    main()
