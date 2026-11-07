"""Surprise SVD helpers for CineMatch Week 3."""

from __future__ import annotations

import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Surprise initializes its built-in dataset paths at import time. Keep that
# cache inside this project rather than writing to the user's home directory.
os.environ.setdefault(
    "SURPRISE_DATA_FOLDER",
    str(Path(__file__).resolve().parents[2] / "work" / "surprise_data"),
)

from surprise import Dataset, Reader, SVD


RATING_SCALE = (0.5, 5.0)


def build_trainset(ratings: pd.DataFrame):
    """Convert a ratings frame to a Surprise full trainset."""
    required = {"userId", "movieId", "rating"}
    if missing := required - set(ratings.columns):
        raise ValueError(f"Missing rating columns: {sorted(missing)}")
    reader = Reader(rating_scale=RATING_SCALE)
    dataset = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    return dataset.build_full_trainset()


def fit_svd(
    ratings: pd.DataFrame,
    n_factors: int,
    *,
    n_epochs: int = 20,
    random_state: int = 42,
) -> SVD:
    """Fit a reproducible biased SVD model."""
    if n_factors < 1:
        raise ValueError("n_factors must be positive")
    algorithm = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        biased=True,
        random_state=random_state,
    )
    algorithm.fit(build_trainset(ratings))
    return algorithm


def predict_frame(algorithm: SVD, ratings: pd.DataFrame) -> pd.DataFrame:
    """Predict every user/movie pair in a ratings frame."""
    rows = []
    for row in ratings.itertuples(index=False):
        prediction = algorithm.predict(int(row.userId), int(row.movieId), r_ui=float(row.rating))
        rows.append(
            {
                "userId": int(row.userId),
                "movieId": int(row.movieId),
                "actual_rating": float(row.rating),
                "predicted_rating": float(prediction.est),
                "was_impossible": bool(prediction.details.get("was_impossible", False)),
            }
        )
    return pd.DataFrame(rows)


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    errors = predictions["predicted_rating"] - predictions["actual_rating"]
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "evaluated_ratings": int(len(predictions)),
        "impossible_predictions": int(predictions["was_impossible"].sum()),
    }


def tune_factors(
    train_ratings: pd.DataFrame,
    validation_ratings: pd.DataFrame,
    factors: tuple[int, ...] = (20, 50, 100),
    *,
    n_epochs: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Select n_factors using validation RMSE only."""
    rows = []
    for n_factors in factors:
        algorithm = fit_svd(
            train_ratings,
            n_factors=n_factors,
            n_epochs=n_epochs,
            random_state=random_state,
        )
        metrics = regression_metrics(predict_frame(algorithm, validation_ratings))
        rows.append({"n_factors": n_factors, **metrics})
    results = pd.DataFrame(rows).sort_values(["rmse", "mae", "n_factors"]).reset_index(drop=True)
    return results, int(results.iloc[0]["n_factors"])


def recommend_svd(
    algorithm: SVD,
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """Score every unseen catalog movie and return the top N."""
    known = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    candidates = movies.loc[~movies["movieId"].isin(known), ["movieId", "title", "genres"]].copy()
    candidates["predicted_rating"] = [
        float(algorithm.predict(user_id, int(movie_id)).est) for movie_id in candidates["movieId"]
    ]
    return candidates.sort_values(["predicted_rating", "movieId"], ascending=[False, True]).head(n).reset_index(drop=True)


def latent_feature_extremes(
    algorithm: SVD,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    factors_to_show: int = 3,
    movies_per_side: int = 5,
    minimum_ratings: int = 20,
) -> pd.DataFrame:
    """Return real movies at both extremes of selected learned item factors.

    Factor signs and order are arbitrary. This output supports exploration but
    does not assign semantic labels such as 'Action vs Romance' automatically.
    """
    counts = ratings.groupby("movieId").size()
    rows = []
    trainset = algorithm.trainset
    eligible = []
    for inner_id in range(trainset.n_items):
        raw_id = int(trainset.to_raw_iid(inner_id))
        if int(counts.get(raw_id, 0)) >= minimum_ratings:
            eligible.append((inner_id, raw_id))
    title_lookup = movies.set_index("movieId")["title"].to_dict()
    for factor in range(min(factors_to_show, algorithm.qi.shape[1])):
        ordered = sorted(eligible, key=lambda pair: algorithm.qi[pair[0], factor])
        selected = [("low", pair) for pair in ordered[:movies_per_side]]
        selected += [("high", pair) for pair in ordered[-movies_per_side:][::-1]]
        for side, (inner_id, movie_id) in selected:
            rows.append(
                {
                    "factor": factor + 1,
                    "side": side,
                    "movieId": movie_id,
                    "title": title_lookup.get(movie_id),
                    "factor_value": float(algorithm.qi[inner_id, factor]),
                    "rating_count": int(counts[movie_id]),
                }
            )
    return pd.DataFrame(rows)


def save_model(algorithm: SVD, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(algorithm, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str | Path) -> SVD:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)
