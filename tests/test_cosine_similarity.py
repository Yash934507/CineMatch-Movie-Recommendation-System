"""Step 1 mathematics, input contracts, and independent sklearn comparisons."""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from cinematch.memory_cf import cosine_similarity, cosine_similarity_manual


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1, 2], [2, 4], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 2], [-2, -4], -1.0),
        ([1, 2], [3, 4], 11 / np.sqrt(125)),
        ([0, 0], [2, 4], 0.0),
        ([2, 4], [0, 0], 0.0),
        ([0, 0], [0, 0], 0.0),
        ([], [], 0.0),
        ([3], [7], 1.0),
    ],
)
def test_known_vectors(a, b, expected):
    assert cosine_similarity_manual(a, b) == pytest.approx(expected)
    assert cosine_similarity(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a,b,message",
    [
        ([1], [1, 2], "same shape"),
        (1, 2, "one-dimensional"),
        ([[1, 2]], [[3, 4]], "one-dimensional"),
        ([np.nan], [1], "finite"),
        ([1], [np.inf], "finite"),
        ([-np.inf], [0], "finite"),
    ],
)
def test_invalid_inputs(a, b, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarity_manual(a, b)


def test_matches_sklearn_on_seeded_vectors():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(12, 20))
    vectors[0] = 0.0
    expected = sklearn_cosine_similarity(vectors)
    actual = np.array([
        [cosine_similarity_manual(a, b) for b in vectors] for a in vectors
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, actual.T, atol=1e-12)
    assert np.all(np.abs(actual) <= 1.0)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_extreme_finite_scales(scale):
    a = np.array([1.0, 2.0]) * scale
    b = np.array([3.0, 4.0]) * scale
    assert cosine_similarity_manual(a, b) == pytest.approx(11 / np.sqrt(125))


def test_inputs_are_not_modified():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    cosine_similarity_manual(a, b)
    np.testing.assert_array_equal(a, [1.0, 2.0])
    np.testing.assert_array_equal(b, [3.0, 4.0])
