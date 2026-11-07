"""MovieLens loading and utility-matrix helpers."""

from pathlib import Path

import pandas as pd

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
MOVIE_COLUMNS = ["movieId", "title", "genres"]


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate MovieLens Small ratings and movies tables."""
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")

    missing_rating_columns = set(RATING_COLUMNS) - set(ratings.columns)
    missing_movie_columns = set(MOVIE_COLUMNS) - set(movies.columns)
    if missing_rating_columns or missing_movie_columns:
        raise ValueError(
            "Unexpected MovieLens schema. "
            f"Missing rating columns: {sorted(missing_rating_columns)}; "
            f"missing movie columns: {sorted(missing_movie_columns)}"
        )
    if ratings.empty or movies.empty:
        raise ValueError("MovieLens input tables must not be empty")
    if ratings[["userId", "movieId", "rating"]].isna().any().any():
        raise ValueError("Ratings contain missing user, movie, or rating values")
    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Ratings must be between 0.5 and 5.0")

    merged = ratings.merge(movies, on="movieId", how="left", validate="many_to_one")
    if merged["title"].isna().any():
        raise ValueError("At least one rating references an unknown movie")
    return ratings, movies, merged


def make_utility_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Return a user-by-movie matrix containing observed ratings and NaN elsewhere."""
    return ratings.pivot(index="userId", columns="movieId", values="rating")


def calculate_sparsity(utility_matrix: pd.DataFrame) -> float:
    """Return the fraction of utility-matrix cells without an observed rating."""
    total_cells = utility_matrix.shape[0] * utility_matrix.shape[1]
    if total_cells == 0:
        raise ValueError("Cannot calculate sparsity for an empty matrix")
    observed = int(utility_matrix.notna().sum().sum())
    return 1.0 - observed / total_cells

