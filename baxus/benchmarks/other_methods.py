import json
import os
from abc import ABC, abstractmethod
from logging import warning
from typing import Tuple, Optional, Dict, Any

import numpy as np

from baxus.benchmarks import Benchmark


class OptimizationMethod(ABC):
    def __init__(
            self,
            run_dir: str,
            conf_name: Optional[str] = None,
    ):
        """
        Abstract base class for a generic optimization method.

        Args:
            run_dir: the directory to store results in
            conf_name: the algorithm configuration to save to disk
        """
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
        if conf_name is not None:
            with open(os.path.join(run_dir, "conf_name.txt"), "w+") as f:
                f.write(conf_name)
        with open(os.path.join(run_dir, "conf_dict.json"), "w+") as f:
            json.dump(self.conf_dict, f)

        self._optimized = False
        self.run_dir = run_dir

    @abstractmethod
    def optimize(self) -> None:
        """
        Start the optimization.

        Returns: None

        """
        raise NotImplementedError()

    @abstractmethod
    def optimization_results_raw(
            self,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Get the raw optimization results, i.e., the x-values, the true function values, and the additional
        run information.

        Returns:
            tuple[X's, y's, additional_run_information]
        """
        raise NotImplementedError()

    def reset(self) -> None:
        warning("No reset implemented.")
        pass

    @property
    def conf_dict(self) -> Dict[str, Any]:
        return {}

    def optimization_results_incumbent(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the incumbent optimization results, i.e., optimization results such that y_2 is always less or equal to
        y_1.

        Returns:
            np.ndarray: the x-values
            np.ndarray: the incumbent y-values

        """
        assert self._optimized, "Model hasn't been optimized yet"
        (
            Xs,
            ys,
        ) = self.optimization_results_raw()
        assert ys.ndim == 1
        ys_incumbent = np.minimum.accumulate(ys)
        return Xs, ys_incumbent


class RandomSearch(OptimizationMethod):
    def __init__(
            self,
            function: Benchmark,
            input_dim: int,
            max_evals: int,
            run_dir: str,
            lower_bounds: np.ndarray,
            upper_bounds: np.ndarray):
        """
        Simple random search implementation, samples points uniformly at random in the search space.

        Args:
            function: the function to optimize
            input_dim: the dimensionality of the problem
            max_evals: maximum number of function evaluations
            run_dir: the directory to save results to
            lower_bounds: the lower bound of the search space
            upper_bounds: the upper_bound of the search space
        """
        super().__init__(run_dir)

        self.run_dir = run_dir

        lower_bounds = np.array(lower_bounds, dtype=np.float32)
        upper_bounds = np.array(upper_bounds, dtype=np.float32)

        assert type(max_evals) == int
        assert type(input_dim) == int
        assert len(lower_bounds) == len(upper_bounds)
        assert len(lower_bounds) == input_dim

        self.function = function
        self.max_evals = max_evals
        self.input_dim = input_dim
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def optimize(self) -> None:
        """
        Run the optimization.

        Returns: None

        """
        assert not self._optimized

        points = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.max_evals, self.input_dim))
        try:
            ys = np.array(self.function(points))
        except:
            warning("Could not run function on all points at once even though"
                    " the function should support this.")
            ys = np.array([self.function(y) for y in points])

        self.ys = ys
        self._optimized = True

    def optimization_results_raw(
            self,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        assert self._optimized, "Model hasn't been optimized yet"
        return None, self.ys
