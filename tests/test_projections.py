from copy import deepcopy
from unittest import TestCase

import numpy as np
import pytest

from baxus.util.exceptions import OutOfBoundsException
from baxus.util.projections import AxUS


class ProjectionsTestSuite(TestCase):
    def test_init(self):
        axus = AxUS(input_dim=57, target_dim=23)
        assert axus.target_dim == 23
        assert axus.input_dim == 57
        assert axus.S_prime.shape == (57, 23)
        assert axus.S.shape == (23, 57)
        for row in axus.S_prime:
            self.assertEqual(1, np.sum(np.abs(row)))

        for col in axus.S.T:
            self.assertEqual(1, np.sum(np.abs(col)))

        self.assertEqual(57, np.count_nonzero(axus.S))
        self.assertEqual(57, np.count_nonzero(axus.S_prime))

    def test_project_down(self):
        axus = AxUS(input_dim=6, target_dim=2)
        axus.S = np.array(
            [[-1, -1, 0, 1, 0, 0], [0, 0, 1, 0, -1, 1]], dtype=np.float32
        )
        Y = axus.project_down(
            np.array([[0.3, 0.7, -0.4, 0.6, -0.1, -0.9]]).T
        ).squeeze()
        assert np.allclose(Y, np.array([-0.4, -1.2]))

    def test_project_up(self):
        axus = AxUS(input_dim=6, target_dim=2)
        axus.S = np.array(
            [[-1, -1, 0, 1, 0, 0], [0, 0, 1, 0, -1, 1]], dtype=np.float32
        )
        Y = axus.project_up(np.array([[-0.4, 0.7]]).T).squeeze()
        assert np.allclose(Y, np.array([0.4, 0.4, 0.7, -0.4, -0.7, 0.7]))

    def test_exceed_bounds_down(self):
        with pytest.raises(OutOfBoundsException):
            axus = AxUS(input_dim=6, target_dim=2)
            axus.project_down(np.array([[0.3, 1.7, -0.4, 0.6, -0.1, -0.9]]).T)

    def test_exceed_bounds_up(self):
        with pytest.raises(OutOfBoundsException):
            axus = AxUS(input_dim=6, target_dim=2)
            axus.project_up(np.array([[-1.4, 0.7]]).T)

    def test_compute_input_to_target_dim(self):
        axus = AxUS(input_dim=40, target_dim=10)
        target_to_input_dim = axus.target_to_input_dim
        input_to_target_dim = axus.input_to_target_dim

        for k, v in input_to_target_dim.items():
            # k: input dim
            # v : target dim
            self.assertIn(k, target_to_input_dim[v])

    def test_reset(self):
        axus0 = AxUS(input_dim=100, target_dim=10)
        axus1 = deepcopy(axus0)
        axus1._reset()

        self.assertEqual(axus0.input_dim, axus1.input_dim)
        self.assertEqual(axus0.target_dim, axus1.target_dim)
        self.assertFalse(np.allclose(axus0.S_prime, axus1.S_prime))
        self.assertFalse(np.allclose(axus0.S, axus1.S))

        ttid0 = axus0.target_to_input_dim
        ttid1 = axus1.target_to_input_dim

        all_equal = True
        for t_dim in range(10):
            np0 = np.array(ttid0[t_dim])
            np1 = np.array(ttid1[t_dim])
            if len(np0) != len(np1):
                all_equal = False
                break
            elif not np.allclose(np0, np1):
                all_equal = False
                break
        self.assertFalse(all_equal)

    def test_bin_distribution(self):
        """
        Test that bins follow the desired almost equal distribution.

        Returns:

        """
        axus = AxUS(input_dim=14, target_dim=4)
        non_zeros_per_row = [np.count_nonzero(axus.S[i]) for i in
                             range(4)]  # there should be two bins with three elements and two bins with four elements
        self.assertEqual(2, len([i for i in non_zeros_per_row if i == 3]))
        self.assertEqual(2, len([i for i in non_zeros_per_row if i == 4]))

    def test_increase_embedding(self):
        axus = AxUS(input_dim=10, target_dim=2)
        row_to_split = axus.S[0].copy()
        axus.increase_target_dimensionality(dims_and_bins={0: 3})  # should increase target dimensionality by 2
        self.assertEqual(axus.S.shape[0], 4)
        self.assertEqual(axus.S_prime.shape[1], 4)
        non_zero_elements_before_split = np.count_nonzero(row_to_split)
        non_zero_in_new_rows = np.count_nonzero(axus.S[-1]) + np.count_nonzero(axus.S[-2])
        self.assertEqual(np.count_nonzero(axus.S[0]), non_zero_elements_before_split - non_zero_in_new_rows)

        self.assertEqual(10, np.count_nonzero(axus.S))
        self.assertEqual(10, np.count_nonzero(axus.S_prime))
