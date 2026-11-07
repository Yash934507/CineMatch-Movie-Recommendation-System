"""CineMatch recommendation-system package."""

from .data import calculate_sparsity, load_movielens, make_utility_matrix
from .evaluation import popularity_fallback, ranking_metrics
from .hybrid import (
    DEFAULT_WEIGHTS,
    build_genre_profile,
    combine_hybrid_scores,
    compute_item_novelty,
    genre_coverage,
    intra_list_diversity,
    mmr_rerank,
    movie_genre_similarity,
    novelty_rerank,
    rank_hybrid_candidates,
    validate_hybrid_weights,
)
from .memory_cf import MemoryBasedCF, cosine_similarity
from .svd_model import fit_svd, recommend_svd

__all__ = [
    "MemoryBasedCF",
    "DEFAULT_WEIGHTS",
    "build_genre_profile",
    "calculate_sparsity",
    "combine_hybrid_scores",
    "compute_item_novelty",
    "cosine_similarity",
    "fit_svd",
    "genre_coverage",
    "intra_list_diversity",
    "load_movielens",
    "make_utility_matrix",
    "mmr_rerank",
    "movie_genre_similarity",
    "novelty_rerank",
    "popularity_fallback",
    "rank_hybrid_candidates",
    "ranking_metrics",
    "recommend_svd",
    "validate_hybrid_weights",
]

