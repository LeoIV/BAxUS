from collections import defaultdict
from functools import partial
from unittest import TestCase

from baxus.benchmarks.synthetic_benchmark_functions import RosenbrockEffectiveDim
from baxus.baxus import BAxUS
import numpy as np


class AdaTheSBO1(TestCase):
    def test_no_progress_on_splits(self):
        split_best = defaultdict(partial(float, np.inf))
        split_best[0] = 32.1
        split_best[1] = 43.6
        split_best[2] = 54.3
        f = RosenbrockEffectiveDim(dim=37)
        adathesbo = BAxUS(
            target_dim=21, n_init=10, max_evals=100, f=f, feature_maps_vae=[16, 8]
        )
        self.assertFalse(
            adathesbo._progress_on_splits(2, split_best=split_best, fail_tolerance=2)
        )
        self.assertEqual(split_best[0], np.inf)
        self.assertEqual(split_best[1], np.inf)
        self.assertEqual(split_best[2], 54.3)
        self.assertTrue(
            adathesbo._progress_on_splits(2, split_best=split_best, fail_tolerance=2)
        )
        self.assertEqual(split_best[0], np.inf)
        self.assertEqual(split_best[1], np.inf)
        self.assertEqual(split_best[2], 54.3)

    def test_progress_on_splits(self):
        split_best = defaultdict(partial(float, np.inf))
        split_best[0] = 65.1
        split_best[1] = 32.6
        split_best[2] = 54.3
        f = RosenbrockEffectiveDim(dim=37)
        adathesbo = BAxUS(
            target_dim=21, n_init=10, max_evals=100, f=f, feature_maps_vae=[16, 8]
        )
        self.assertTrue(
            adathesbo._progress_on_splits(2, split_best=split_best, fail_tolerance=2)
        )
        self.assertFalse(
            adathesbo._progress_on_splits(2, split_best=split_best, fail_tolerance=1)
        )
        self.assertEqual(split_best[1], np.inf)
        self.assertEqual(split_best[0], 65.1)

    def test_join_data(self):
        f = RosenbrockEffectiveDim(2)
        at = BAxUS(
            target_dim=2, n_init=10, max_evals=100, f=f, feature_maps_vae=[16, 8]
        )
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x1 = at._join_data(X=x, splitting_dim=0, n_new_bins=2)
        self.assertTrue(np.allclose(x1[:, 0], x1[:, 3]))
        self.assertTrue(np.allclose(x[:, 0], x1[:, 3]))
        self.assertTrue(np.allclose(x[:, 0], x1[:, 0]))
        self.assertTrue(np.allclose(x[:, 1], x1[:, 1]))
        self.assertTrue(np.allclose(x[:, 2], x1[:, 2]))
        self.assertTrue(x1.shape == (2, 4))
        self.assertTrue(x.shape == (2, 3))

    def test_choose_splitting_dim(self):
        f = RosenbrockEffectiveDim(dim=40)
        at = BAxUS(
            target_dim=10, n_init=10, max_evals=100, f=f, feature_maps_vae=[16, 8]
        )
        target_to_input_dim = {i: list(range(4 * i, 4 * (i + 1))) for i in range(10)}
        at.projector.target_to_input_dim = target_to_input_dim
        at.projector.S_prime = at.projector._compute_proj_mtrx(
            target_dim=10,
            input_dim=40,
            input_dim_to_sign_sigma=at.projector.input_dim_to_sign_sigma,
            target_to_input_dim=target_to_input_dim,
        )
        for i in range(10):
            at.lengthscales = np.random.uniform(low=0.1, high=1.0, size=10)
            at.lengthscales[i] = 0.05
            chosen_sd, _ = at._choose_splitting_dim(
                projector=at.projector,
                lengthscales=at.lengthscales,
                probabilistic=False,
            )
            self.assertEqual(chosen_sd, i)

    def test_choose_splitting_dim_probab(self):
        f = RosenbrockEffectiveDim(dim=40)
        at = BAxUS(
            target_dim=10, n_init=10, max_evals=100, f=f, feature_maps_vae=[16, 8]
        )
        target_to_input_dim = {i: list(range(4 * i, 4 * (i + 1))) for i in range(10)}
        at.projector.target_to_input_dim = target_to_input_dim
        at.projector.S_prime = at.projector._compute_proj_mtrx(
            target_dim=10,
            input_dim=40,
            input_dim_to_sign_sigma=at.projector.input_dim_to_sign_sigma,
            target_to_input_dim=target_to_input_dim,
        )
        for i in range(10):
            idxs = np.zeros(10, dtype=np.int32)
            for _ in range(10000):
                at.lengthscales = np.random.uniform(low=0.1, high=1.0, size=10)
                at.lengthscales[i] = 0.05
                chosen_sd = at._choose_splitting_dim(
                    projector=at.projector,
                    lengthscales=at.lengthscales,
                    probabilistic=True,
                )
                idxs[chosen_sd] += 1
            self.assertEqual(np.argmax(idxs), i)
