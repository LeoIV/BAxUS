import math
import os
from typing import Optional

import numpy as np
import torch
from botorch.test_functions import Ackley as BotorchAckley
from botorch.test_functions import Branin as BotorchBranin
from botorch.test_functions import DixonPrice as BotorchDixonPrice
from botorch.test_functions import Griewank as BotorchGriewank
from botorch.test_functions import Hartmann as BotorchHartmann
from botorch.test_functions import Levy as BotorchLevy
from botorch.test_functions import Michalewicz as BotorchMichalewicz
from botorch.test_functions import Rastrigin as BotorchRastrigin
from botorch.test_functions import Rosenbrock as BotorchRosenbrock

from baxus.benchmarks import EffectiveDimBoTorchBenchmark


class AckleyEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    A benchmark function with many local minima (see https://www.sfu.ca/~ssurjano/ackley.html)

    WARNING: This function has its optimum at the origin. This might give a misleading performance for BAxUS
    as the origin will always be reachable irregardless of the embedding.

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim=200, noise_std=None, effective_dim: int = 10):
        super(AckleyEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=np.full(shape=effective_dim, fill_value=-32.768),
            ub=np.full(shape=effective_dim, fill_value=32.768),
            benchmark_func=BotorchAckley,
        )


class ShiftedAckley10(EffectiveDimBoTorchBenchmark):
    """
    A benchmark function with many local minima (see https://www.sfu.ca/~ssurjano/ackley.html)

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim=200, noise_std=None, ):
        self.offsets = np.array([-14.15468831, -17.35934204, 4.93227439, 30.68108305,
                                 -20.94097318, -9.68946759, 11.23919487, 4.93101114,
                                 2.87604112, -31.0805155])

        super(ShiftedAckley10, self).__init__(
            dim=dim,
            effective_dim=10,
            noise_std=noise_std,
            lb=np.full(shape=10, fill_value=-32.768) - self.offsets,
            ub=np.full(shape=10, fill_value=32.768) - self.offsets,
            benchmark_func=BotorchAckley,
        )

    def __call__(self, x):
        x = np.array(x)
        return super().__call__(x, self.offsets)


class RosenbrockEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    A valley-shape benchmark function (see https://www.sfu.ca/~ssurjano/rosen.html)

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(
            self, dim: int = 200, noise_std: Optional[float] = None, effective_dim: int = 10
    ):
        super().__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            ub=np.full(shape=effective_dim, fill_value=10),
            lb=np.full(shape=effective_dim, fill_value=-5),
            benchmark_func=BotorchRosenbrock,
        )


class HartmannEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    A valley-shape benchmark function (see https://www.sfu.ca/~ssurjano/rosen.html)

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(
            self, dim: int = 200, noise_std: Optional[float] = None, effective_dim: int = 6
    ):
        assert effective_dim == 6
        super().__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            ub=np.ones(effective_dim),
            lb=np.zeros(effective_dim),
            benchmark_func=BotorchHartmann,
        )


class BraninEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Branin function with three local minima (see https://www.sfu.ca/~ssurjano/branin.html)

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(
            self, dim: int = 200, noise_std: Optional[float] = None, effective_dim: int = 2
    ):
        assert effective_dim == 2
        super().__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=np.array([-5.0, -5.0]),
            ub=np.array([15.0, 15.0]),
            benchmark_func=BotorchBranin,
        )


class LevyEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Levy function with many local minima (see https://www.sfu.ca/~ssurjano/levy.html)

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim=200, noise_std=None, effective_dim: int = 2):
        super(LevyEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=np.full(shape=effective_dim, fill_value=-10),
            ub=np.full(shape=effective_dim, fill_value=10),
            benchmark_func=BotorchLevy,
        )


class DixonPriceEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The valley shaped Dixon-Price function (see https://www.sfu.ca/~ssurjano/dixonpr.html)

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim=200, noise_std=None, effective_dim: int = 2):
        super(DixonPriceEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=np.full(shape=effective_dim, fill_value=-10),
            ub=np.full(shape=effective_dim, fill_value=10),
            benchmark_func=BotorchDixonPrice,
        )


class GriewankEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Griewank function with many local minima (see https://www.sfu.ca/~ssurjano/griewank.html)

    WARNING: This function has its optimum at the origin. This might give a misleading performance for BAxUS
    as the origin will always be reachable irregardless of the embedding.

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim=200, noise_std=None, effective_dim: int = 2):
        super(GriewankEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=np.full(shape=effective_dim, fill_value=-600),
            ub=np.full(shape=effective_dim, fill_value=600),
            benchmark_func=BotorchGriewank,
        )


class MichalewiczEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Michalewicz function with steep drops (see https://www.sfu.ca/~ssurjano/michal.html)

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim=200, noise_std=None, effective_dim: int = 2):
        super(MichalewiczEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=np.full(shape=effective_dim, fill_value=0),
            ub=np.full(shape=effective_dim, fill_value=math.pi),
            benchmark_func=BotorchMichalewicz,
        )


class RastriginEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Rastrigin function with many local minima (see https://www.sfu.ca/~ssurjano/rastr.html)

    WARNING: This function has its optimum at the origin. This might give a misleading performance for BAxUS
    as the origin will always be reachable irregardless of the embedding.

    Args:
        dim: The ambient dimensionality of the function
        noise_std: The standard deviation of the noise
        effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim=200, noise_std=None, effective_dim: int = 2):
        super(RastriginEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=np.full(shape=effective_dim, fill_value=-5.12),
            ub=np.full(shape=effective_dim, fill_value=5.12),
            benchmark_func=BotorchRastrigin,
        )


class RotatedHartmann6(EffectiveDimBoTorchBenchmark):
    """
    Version of the rotated Hartmann6 function as described in https://bit.ly/3dZFVXv

    Args:
        noise_std: The standard deviation of the noise
    """

    def __init__(self, noise_std: Optional[float] = None, **kwargs):
        # bounds taken from https://bit.ly/3e0YgDw
        super().__init__(1000, noise_std, 6, np.ones(6), np.zeros(6), BotorchHartmann)
        # this is the same matrix as in rotation_matrix_alebo.json
        self.rotation_matrix = np.load(
            os.path.join("data", "rotation_matrix_alebo.npy")
        )

    def __call__(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        else:
            assert x.ndim == 2
        x = x.T
        x_r = self.rotation_matrix @ x
        res = self._benchmark_func.forward(torch.tensor(x_r.T)).numpy().squeeze()
        return res
