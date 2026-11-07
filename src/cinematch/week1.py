"""Week 1 exploratory analysis and artifact generation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .data import calculate_sparsity, load_movielens, make_utility_matrix


def build_summary(ratings: pd.DataFrame, movies: pd.DataFrame, utility: pd.DataFrame) -> dict:
    counts = ratings["rating"].value_counts().sort_index()
    popularity = ratings.groupby("movieId").size().sort_values(ascending=False)
    return {
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "movies_in_catalog": int(len(movies)),
        "movies_with_ratings": int(ratings["movieId"].nunique()),
        "utility_rows": int(utility.shape[0]),
        "utility_columns": int(utility.shape[1]),
        "observed_cells": int(utility.notna().sum().sum()),
        "total_cells": int(utility.size),
        "sparsity_ratio": calculate_sparsity(utility),
        "mean_rating": float(ratings["rating"].mean()),
        "median_rating": float(ratings["rating"].median()),
        "ratings_4_or_5_share": float(ratings["rating"].isin([4.0, 5.0]).mean()),
        "rating_counts": {str(k): int(v) for k, v in counts.items()},
        "top_10_movies_by_rating_count": [
            {"movieId": int(movie_id), "title": str(title), "rating_count": int(count)}
            for movie_id, title, count in (
                popularity.head(10).rename("rating_count").reset_index().merge(
                    movies[["movieId", "title"]], on="movieId", how="left"
                )[["movieId", "title", "rating_count"]].itertuples(index=False, name=None)
            )
        ],
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size=size)
    except OSError:
        return ImageFont.load_default()


def _canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((45, 35, 1555, 865), radius=24, fill="white", outline="#e2e8f0", width=2)
    draw.text((95, 75), title, fill="#0f172a", font=_font(42, bold=True))
    if subtitle:
        draw.text((95, 130), subtitle, fill="#64748b", font=_font(23))
    return image, draw


def _save_rating_distribution(ratings: pd.DataFrame, path: Path) -> None:
    counts = ratings["rating"].value_counts().sort_index()
    image, draw = _canvas("MovieLens Small rating distribution", "Count of ratings at each half-star value")
    left, top, right, bottom = 150, 220, 1500, 760
    draw.line((left, top, left, bottom), fill="#94a3b8", width=3)
    draw.line((left, bottom, right, bottom), fill="#94a3b8", width=3)
    max_count = int(counts.max())
    slot = (right - left) / len(counts)
    for i, (rating, count) in enumerate(counts.items()):
        height = (count / max_count) * (bottom - top - 50)
        x0 = left + i * slot + 18
        x1 = left + (i + 1) * slot - 18
        y0 = bottom - height
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=9, fill="#7c3aed")
        label = f"{rating:g}"
        draw.text(((x0 + x1) / 2, bottom + 18), label, anchor="ma", fill="#334155", font=_font(22))
        draw.text(((x0 + x1) / 2, y0 - 12), f"{count:,}", anchor="ms", fill="#334155", font=_font(18, bold=True))
    draw.text(((left + right) / 2, 825), "Rating", anchor="mm", fill="#334155", font=_font(24, bold=True))
    image.save(path)


def _save_long_tail(popularity: pd.DataFrame, path: Path) -> None:
    image, draw = _canvas("Long tail of movie popularity", "Both axes use logarithmic scales")
    left, top, right, bottom = 150, 215, 1500, 765
    draw.line((left, top, left, bottom), fill="#94a3b8", width=3)
    draw.line((left, bottom, right, bottom), fill="#94a3b8", width=3)
    max_rank = len(popularity)
    max_count = int(popularity["rating_count"].max())
    points = []
    for rank, count in zip(popularity["rank"], popularity["rating_count"]):
        x = left + math.log10(rank) / math.log10(max_rank) * (right - left)
        y = bottom - math.log10(count) / math.log10(max_count) * (bottom - top)
        points.append((x, y))
    draw.line(points, fill="#2563eb", width=4)
    for exponent in range(0, math.ceil(math.log10(max_rank)) + 1):
        value = 10**exponent
        if value <= max_rank:
            x = left + math.log10(value) / math.log10(max_rank) * (right - left)
            draw.line((x, bottom, x, bottom + 8), fill="#64748b", width=2)
            draw.text((x, bottom + 18), f"{value:,}", anchor="ma", fill="#475569", font=_font(20))
    for value in [1, 10, 100, max_count]:
        if value <= max_count:
            y = bottom - math.log10(value) / math.log10(max_count) * (bottom - top)
            draw.line((left - 8, y, right, y), fill="#e2e8f0", width=2)
            draw.text((left - 18, y), f"{value:,}", anchor="rm", fill="#475569", font=_font(20))
    draw.text(((left + right) / 2, 825), "Movie rank by number of ratings", anchor="mm", fill="#334155", font=_font(24, bold=True))
    image.save(path)


def _save_top_movies(popularity: pd.DataFrame, path: Path) -> None:
    top_movies = popularity.head(15).iloc[::-1]
    image, draw = _canvas("15 most-rated movies", "Popularity is the number of observed ratings")
    left, top, right, bottom = 560, 195, 1500, 810
    row_height = (bottom - top) / len(top_movies)
    max_count = int(top_movies["rating_count"].max())
    for i, row in enumerate(top_movies.itertuples(index=False)):
        y0 = top + i * row_height + 5
        y1 = top + (i + 1) * row_height - 5
        width = row.rating_count / max_count * (right - left)
        title = row.title if len(row.title) <= 43 else row.title[:40] + "..."
        draw.text((left - 18, (y0 + y1) / 2), title, anchor="rm", fill="#334155", font=_font(18))
        draw.rounded_rectangle((left, y0, left + width, y1), radius=7, fill="#0f766e")
        draw.text((left + width + 12, (y0 + y1) / 2), f"{row.rating_count}", anchor="lm", fill="#334155", font=_font(18, bold=True))
    image.save(path)


def save_plots(ratings: pd.DataFrame, movies: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    popularity = (
        ratings.groupby("movieId").size().rename("rating_count").reset_index()
        .merge(movies[["movieId", "title"]], on="movieId", how="left")
        .sort_values("rating_count", ascending=False)
        .reset_index(drop=True)
    )
    popularity["rank"] = popularity.index + 1
    _save_rating_distribution(ratings, figures_dir / "rating_distribution.png")
    _save_long_tail(popularity, figures_dir / "long_tail_distribution.png")
    _save_top_movies(popularity, figures_dir / "top_15_most_rated.png")


def run_week1(project_root: str | Path) -> dict:
    root = Path(project_root)
    ratings, movies, merged = load_movielens(root / "data" / "raw")
    utility = make_utility_matrix(ratings)
    summary = build_summary(ratings, movies, utility)

    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    merged.to_csv(processed / "ratings_with_movies.csv", index=False)
    utility.to_pickle(processed / "utility_matrix.pkl")
    popularity = (
        merged.groupby(["movieId", "title"], as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
        .sort_values(["rating_count", "mean_rating"], ascending=False)
    )
    popularity.to_csv(processed / "movie_popularity.csv", index=False)
    (processed / "week1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_plots(ratings, movies, root / "reports" / "figures")
    return summary

