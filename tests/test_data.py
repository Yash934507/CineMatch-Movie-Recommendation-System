import pandas as pd
import unittest

from cinematch.data import calculate_sparsity, make_utility_matrix


class DataTests(unittest.TestCase):
    def test_utility_matrix_shape_and_values(self):
        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 2],
                "movieId": [10, 20, 10],
                "rating": [4.0, 3.5, 5.0],
            }
        )
        matrix = make_utility_matrix(ratings)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.loc[1, 20], 3.5)
        self.assertTrue(pd.isna(matrix.loc[2, 20]))

    def test_sparsity_uses_missing_cells(self):
        matrix = pd.DataFrame([[4.0, None], [5.0, None]])
        self.assertAlmostEqual(calculate_sparsity(matrix), 0.5)

    def test_empty_matrix_has_no_defined_sparsity(self):
        with self.assertRaises(ValueError):
            calculate_sparsity(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()

