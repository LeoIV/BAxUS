from abc import ABC, abstractmethod
from logging import info
from typing import Optional, Union, List, Type

import numpy as np
import torch
from botorch.test_functions import SyntheticTestFunction

from baxus.util.exceptions import EffectiveDimTooLargeException, BoundsMismatchException, OutOfBoundsException


class Benchmark(ABC):
    """
    Abstract benchmark function.

    Args:
        dim: dimensionality of the objective function
        noise_std: the standard deviation of the noise (None means no noise)
        ub: the upper bound, the object will have the attribute ub_vec which is an np array of length dim filled with ub
        lb: the lower bound, the object will have the attribute lb_vec which is an np array of length dim filled with lb
        benchmark_func: the benchmark function, should inherit from SyntheticTestFunction
    """

    def __init__(self, dim: int, ub: np.ndarray, lb: np.ndarray, noise_std: float):

        lb = np.array(lb)
        ub = np.array(ub)
        if (
                not lb.shape == ub.shape
                or not lb.ndim == 1
                or not ub.ndim == 1
                or not dim == len(lb) == len(ub)
        ):
            raise BoundsMismatchException()
        if not np.all(lb < ub):
            raise OutOfBoundsException()
        self.noise_std = noise_std
        self._dim = dim
        self._lb_vec = lb.astype(np.float32)
        self._ub_vec = ub.astype(np.float32)

    @property
    def dim(self) -> int:
        """
        The benchmark dimensionality

        Returns: the benchmark dimensionality

        """
        return self._dim

    @property
    def lb_vec(self) -> np.ndarray:
        """
        The lower bound of the search space of this benchmark (length = benchmark dim)

        Returns: The lower bound of the search space of this benchmark (length = benchmark dim)

        """
        return self._lb_vec

    @property
    def ub_vec(self) -> np.ndarray:
        """
        The upper bound of the search space of this benchmark (length = benchmark dim)

        Returns: The upper bound of the search space of this benchmark (length = benchmark dim)

        """
        return self._ub_vec

    @property
    def fun_name(self) -> str:
        """
        The name of the benchmark function

        Returns: The name of the benchmark function

        """
        return self.__class__.__name__

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        raise NotImplementedError()


class SyntheticBenchmark(Benchmark):
    """
    Abstract class for synthetic benchmarks

    Args:
        dim: the benchmark dimensionality
        ub: np.ndarray: the upper bound of the search space of this benchmark (length = benchmark dim)
        lb: np.ndarray: the lower bound of the search space of this benchmark (length = benchmark dim)
    """

    @abstractmethod
    def __init__(self, dim: int, ub: np.ndarray, lb: np.ndarray, noise_std: float):
        super().__init__(dim, ub, lb, noise_std=noise_std)

    @abstractmethod
    def __call__(
            self, x: Union[np.ndarray, List[float], List[List[float]]]
    ) -> np.ndarray:
        """
        Call the benchmark function for one or multiple points.

        Args:
            x: Union[np.ndarray, List[float], List[List[float]]]: the x-value(s) to evaluate. numpy array can be 1 or 2-dimensional

        Returns:
            np.ndarray: The function values.


        """
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        # for y in x:
        #    if not np.sum(y < self._lb_vec) == 0:
        #        raise OutOfBoundsException()
        #    if not np.sum(y > self._ub_vec) == 0:
        #        raise OutOfBoundsException

    @property
    def optimal_value(self) -> Optional[np.ndarray]:
        """

        Returns:
            Optional[Union[float, np.ndarray]]: the optimal value if known

        """
        return None


class EffectiveDimBenchmark(SyntheticBenchmark):
    """
    A benchmark with a known effective dimensionality.

    .. note::
        This is an abstract class that needs an implementation.


    Args:
        dim: the overall dimensionality of the problem
        effective_dim: the effective dimensionality of the problem
        ub: the upper bounds of the search space
        lb: the lower bounds of the search space
        noise_std: the noise std for this benchmark
    """

    def __init__(
            self,
            dim: int,
            effective_dim: int,
            ub: np.ndarray,
            lb: np.ndarray,
            noise_std: float,
    ):
        super().__init__(dim, ub, lb, noise_std=noise_std)
        self.effective_dim: int = effective_dim

    @abstractmethod
    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        raise NotImplementedError()


class BoTorchFunctionBenchmark(SyntheticBenchmark):
    """
    A benchmark function that calls a BoTorch benchmark function

    Args:
        dim: dimensionality of the problem
        noise_std: noise std of the function
        ub: the upper bound of the search space
        lb: the lower bound of the search space
        benchmark_func: the BoTorch benchmark function
    """

    def __init__(
            self,
            dim: int,
            noise_std: Optional[float],
            ub: np.ndarray,
            lb: np.ndarray,
            benchmark_func: Type[SyntheticTestFunction],
    ):
        super().__init__(dim, ub=ub, lb=lb, noise_std=noise_std)
        try:
            self._benchmark_func = benchmark_func(dim=dim, noise_std=noise_std)
        except:
            self._benchmark_func = benchmark_func(noise_std=noise_std)

    @property
    def effective_dim(self) -> int:
        """
        The effective dimensionality of the benchmark.

        Returns: The effective dimensionality of the benchmark.

        """
        return self._dim

    @property
    def optimal_value(self) -> np.ndarray:
        """
        The optimal value of the benchmark function.

        Returns: The optimal value of the benchmark function.

        """
        return self._benchmark_func.optimal_value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call the function

        Args:
            x: points at which to evaluate the function

        Returns: function value(s)

        """
        super(BoTorchFunctionBenchmark, self).__call__(x)
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        res = (
            self._benchmark_func.forward(
                torch.tensor(np.clip(x, self._lb_vec, self._ub_vec))
            )
            .numpy()
            .squeeze()
        )
        return res


class EffectiveDimBoTorchBenchmark(BoTorchFunctionBenchmark):
    """
    A benchmark class for synthetic benchmarks with a known effective dimensionality that are based on a BoTorch
    implementation.

    Args:
        dim: int: the ambient dimensionality of the benchmark
        noise_std: float: standard deviation of the noise of the benchmark function
        effective_dim: int: the desired effective dimensionality of the benchmark function
        ub: np.ndarray: the upper bound of the benchmark search space. length = dim
        lb: np.ndarray: the lower bound of the benchmark search space. length = dim
        benchmark_func: Type[SyntheticTestFunction]: the BoTorch benchmark function to use
        seed: int: random seed
    """

    def __init__(
            self,
            dim: int,
            noise_std: Optional[float],
            effective_dim: int,
            ub: np.ndarray,
            lb: np.ndarray,
            benchmark_func: Type[SyntheticTestFunction],
            seed: int = 123,
    ):
        super().__init__(
            effective_dim, noise_std, ub=ub, lb=lb, benchmark_func=benchmark_func
        )
        if effective_dim > dim:
            raise EffectiveDimTooLargeException()
        state = np.random.get_state()
        np.random.seed(seed)
        self._fake_dim = dim
        self._effective_dim = effective_dim
        self.effective_dims = np.arange(dim)[:effective_dim]
        info(f"effective dims: {list(self.effective_dims)}")
        np.random.set_state(state)

    def __call__(self, x, offset: np.ndarray = None) -> np.ndarray:
        """
        Call the function

        Args:
            x: points at which to evaluate the function
            offset: offset to add to the x values (we used this for our ablation study with ShiftedAckley10)

        Returns: the function value(s)

        """
        if offset is None:
            res = super(EffectiveDimBoTorchBenchmark, self).__call__(
                x.squeeze().T[self.effective_dims].T
            )
        else:
            res = super(EffectiveDimBoTorchBenchmark, self).__call__(
                x.squeeze().T[self.effective_dims].T + offset
            )
        return res

    @property
    def dim(self):
        """
        The dimensionality of the problem.

        Returns: The dimensionality of the problem.

        """
        return self._fake_dim

    @property
    def effective_dim(self) -> int:
        """
        the effective dimensionality of the benchmark

        Returns: the effective dimensionality of the benchmark

        """
        return self._effective_dim

    @property
    def lb_vec(self) -> np.ndarray:
        """
        The lower bounds of the search space.

        Returns: The lower bounds of the search space.

        """
        return np.full(
            shape=self._fake_dim, fill_value=np.min(self._lb_vec), dtype=np.float32
        )

    @property
    def ub_vec(self) -> np.ndarray:
        """
        The upper bounds of the search space.

        Returns: The upper bounds of the search space.

        """
        return np.full(
            shape=self._fake_dim, fill_value=np.max(self._ub_vec), dtype=np.float32
        )
