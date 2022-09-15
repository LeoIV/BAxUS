from copy import deepcopy
from unittest import TestCase

import numpy as np
import pytest

from baxus.util.exceptions import OutOfBoundsException
from baxus.util.projections import AxUS


class HeSBOTestCase(TestCase):
    def test_init(self):
        hesbo = AxUS(input_dim=57, target_dim=23)
        assert hesbo.target_dim == 23
        assert hesbo.input_dim == 57
        assert hesbo.S_prime.shape == (57, 23)
        assert hesbo.S.shape == (23, 57)
        for row in hesbo.S_prime:
            assert np.sum(np.abs(row)) == 1

        for col in hesbo.S.T:
            assert np.sum(np.abs(col)) == 1

    def test_compute_projection_matrix(self):
        hesbo = AxUS(input_dim=6, target_dim=2)
        ST = hesbo._compute_proj_mtrx(
            2, 6, {0: -1, 1: -1, 2: 1, 3: 1, 4: -1, 5: 1}, {0: [0, 1, 3], 1: [2, 4, 5]}
        )
        assert np.array_equal(
            ST,
            np.array([[-1, -1, 0, 1, 0, 0], [0, 0, 1, 0, -1, 1]], dtype=np.float32).T,
        )

    def test_project_down(self):
        hesbo = AxUS(input_dim=6, target_dim=2)
        hesbo.S_prime = np.array(
            [[-1, -1, 0, 1, 0, 0], [0, 0, 1, 0, -1, 1]], dtype=np.float32
        ).T
        Y = hesbo.project_down(
            np.array([[0.3, 0.7, -0.4, 0.6, -0.1, -0.9]]).T
        ).squeeze()
        assert np.allclose(Y, np.array([-0.4, -1.2]))

    def test_project_up(self):
        hesbo = AxUS(input_dim=6, target_dim=2)
        hesbo.S_prime = np.array(
            [[-1, -1, 0, 1, 0, 0], [0, 0, 1, 0, -1, 1]], dtype=np.float32
        ).T
        Y = hesbo.project_up(np.array([[-0.4, 0.7]]).T).squeeze()
        assert np.allclose(Y, np.array([0.4, 0.4, 0.7, -0.4, -0.7, 0.7]))

    def test_exceed_bounds_down(self):
        with pytest.raises(OutOfBoundsException):
            hesbo = AxUS(input_dim=6, target_dim=2)
            hesbo.project_down(np.array([[0.3, 1.7, -0.4, 0.6, -0.1, -0.9]]).T)

    def test_exceed_bounds_up(self):
        with pytest.raises(OutOfBoundsException):
            hesbo = AxUS(input_dim=6, target_dim=2)
            hesbo.project_up(np.array([[-1.4, 0.7]]).T)

    def test_compute_input_to_target_dim(self):
        hesbo = AxUS(input_dim=40, target_dim=10)
        target_to_input_dim = hesbo.target_to_input_dim
        input_to_target_dim = hesbo._input_to_target_dim(target_to_input_dim)

        for k, v in input_to_target_dim.items():
            # k: input dim
            # v : target dim
            self.assertIn(k, target_to_input_dim[v])

    def test_reset(self):
        hesbo0 = AxUS(input_dim=100, target_dim=10)
        hesbo1 = deepcopy(hesbo0)
        hesbo1._reset()

        self.assertEqual(hesbo0.input_dim, hesbo1.input_dim)
        self.assertEqual(hesbo0.target_dim, hesbo1.target_dim)
        self.assertFalse(np.allclose(hesbo0.S_prime, hesbo1.S_prime))
        self.assertFalse(np.allclose(hesbo0.S, hesbo1.S))

        ttid0 = hesbo0.target_to_input_dim
        ttid1 = hesbo1.target_to_input_dim

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
