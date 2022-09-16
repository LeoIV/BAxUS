###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

# Derived from the TuRBO implementation (https://github.com/uber-research/TuRBO)
# Author: Leonard Papenmeier <leonard.papenmeier@cs.lth.se>

import base64
import math
import pickle
from copy import deepcopy
from logging import info, debug, warning
from typing import Dict, Optional

import numpy as np

from baxus import EmbeddedTuRBO
from baxus.benchmarks.benchmark_function import Benchmark
from baxus.util.behaviors import BaxusBehavior
from baxus.util.behaviors.gp_configuration import GPBehaviour
from baxus.util.data_utils import join_data
from baxus.util.projections import AxUS, ProjectionModel
from baxus.util.utils import (
    one_around_origin_latin_hypercube,
    from_1_around_origin,
    star_string,
)


class BAxUS(EmbeddedTuRBO):
    """
    BAxUS main class.

    Args:
        f: the function to optimize
        target_dim: the latent dimensionality
        n_init: number of initial samples
        max_evals: max number of function evaluations
        behavior: behavior configuration
        gp_behaviour: the behavior of the associated Gaussian Process
        verbose: verbose logging model
        use_ard: whether the GP should use an ARD kernel (yes this should be part of the gp_behavior)
        max_cholesky_size:
        dtype: the datatype (float32, float64)
        run_dir: the directory to which to write the run results
        conf_name: the name of the configuration of the optimization run
    """

    def __init__(
            self,
            f: Benchmark,
            target_dim: int,
            n_init: int,
            max_evals: int,
            behavior: BaxusBehavior = BaxusBehavior(),
            gp_behaviour: GPBehaviour = GPBehaviour(),
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            dtype="float64",
            run_dir=".",
            conf_name: Optional[str] = None,
    ):
        self.behavior = behavior
        # need to set this here, so we can adjust the initial target dim before initializing super()
        self._input_dim = f.dim
        self._init_target_dim = target_dim
        if self.behavior.adjust_initial_target_dim:
            target_dim = self._adjust_init_target_dim()
            self._init_target_dim = target_dim

        super().__init__(
            f=f,
            target_dim=target_dim,
            n_init=n_init,
            max_evals=max_evals,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            gp_behaviour=gp_behaviour,
            dtype=dtype,
            run_dir=run_dir,
            behavior=behavior,
            conf_name=conf_name,
        )
        self._target_dim_after_reset = self.target_dim
        assert (
                self.length_init > 2 * self.length_min
        ), f"Initial length {self.length_init} has to be larger than two times the minimum length {self.length_min}."

        self._axus_change_iterations = []
        self._split_points = []
        self._trust_region_restarts = []
        self._dim_in_iterations = {}

        self._data_dims = []

    @property
    def target_dim_increases(self) -> int:
        """
        Returns the number of times the target dimensionality was increased.
        This is not the current target dimensionality minus the initial target dimensionality.

        Returns: The number of times the target dimensionality was increased.

        """
        base = self.behavior.n_new_bins + 1
        return round(math.log(self.target_dim / self._init_target_dim, base))

    @EmbeddedTuRBO.target_dim.setter
    def target_dim(self, target_dim: int) -> None:
        """
        Setter for target dimensionality.

        Args:
            target_dim: the new target dimensionality

        Returns: None

        """
        self._dim_in_iterations[self.n_evals] = target_dim
        self._target_dim = target_dim

    @property
    def splits(self) -> int:
        """
        The number of splits in the current trust region.

        Returns: The number of splits in the current trust region.

        """
        base = self.behavior.n_new_bins + 1
        return round(math.log(self.target_dim / self._target_dim_after_reset, base))

    @property
    def length_min(self) -> float:
        """
        The minimum base length of the trust region.

        Returns: The minimum base length of the trust region.

        """
        return self._length_min

    @property
    def length_max(self) -> float:
        """
        The maximum base length of the trust region.

        Returns: The maximum base length of the trust region.

        """
        return self._length_max

    @property
    def length_init(self) -> float:
        """
        The initial base length of the trust region.

        Returns: The initial base length of the trust region.

        """
        return self._length_init

    @property
    def evaluations_since_last_split(self) -> int:
        """
        The number of function evaluations since the last split.

        Returns: The number of function evaluations since the last split. Total number of evaluations if there was no split yet.

        """
        return (
            self.n_evals - self._axus_change_iterations[-1]
            if len(self._axus_change_iterations) > 0
            else self.n_evals
        )

    @property
    def _evaluations_in_last_splits_in_tr(self) -> int:
        """
        The evaluations spent in previous splits in the current trust region

        Returns: the evaluations spent in previous splits in the current trust region

        """
        split_points = np.array(self._split_points)
        if len(self._trust_region_restarts) > 0:
            split_points = split_points[split_points >= self._trust_region_restarts[-1]]
        if len(split_points) == 0:
            return 0
        else:
            if len(self._trust_region_restarts) == 0:
                return split_points[-1]
            else:
                return split_points[-1] - self._trust_region_restarts[-1]

    @property
    def _dimension_importances(self) -> np.ndarray:
        """
        The (inverse) dimension importances. This just returns the lengthscales of the GP ARD kernel.

        Returns: The (inverse) dimension importances. This just returns the lengthscales of the GP ARD kernel.

        """
        return np.array(self.lengthscales)

    @property
    def _split_in_trust_region(self) -> int:
        """
        The number of this split in the current trust region, i.e., if we just reset the trust region and haven't
        split yet, this is 1. Then, after the first split, 2, etc.

        Returns: the number of this split

        """
        if len(self._trust_region_restarts) == 0:
            # if the trust region was not yet restarted, just return the number of splits
            return len(self._split_points) + 1
        else:
            iteration_of_restart = self._trust_region_restarts[-1]
            sp = np.array(self._split_points)
            return len(sp[sp >= iteration_of_restart]) + 1

    @property
    def _init_dim_in_tr(self) -> int:
        """
        The dim with which the current trust region started.

        Returns: The dim with which the current trust region started.

        """
        dim_in_iterations = self._dim_in_iterations
        if len(dim_in_iterations) == 0:
            # target dim was not yet adjusted
            return self._init_target_dim
        else:
            eval_when_tr_started = 0 if len(self._trust_region_restarts) == 0 else self._trust_region_restarts[-1]
            tr_adjust_iters = np.array(list(dim_in_iterations.keys()))
            min_iter = min(tr_adjust_iters[tr_adjust_iters >= eval_when_tr_started])
            return self._dim_in_iterations[min_iter]

    @property
    def _budget_lost_in_previous_trs(self) -> int:
        """
        The number of function evaluations used in previous trust regions.

        Returns: The number of function evaluations used in previous trust regions.

        """
        return self.n_init if len(self._trust_region_restarts) == 0 else self._trust_region_restarts[-1]

    def _adjust_init_target_dim(self) -> int:
        """
        Adjust the initial target dimension such that the final target dimension
        is as close to the ambient dimensionality as possible given a fixed b.

        Returns: int: the adjusted initial target dimension.

        """

        def ndiff(b, d0):
            psi = 1
            desired_final_dim = self.input_dim
            initial_target_dim = d0

            base = psi * b + 1
            n = round(math.log(desired_final_dim / initial_target_dim, base))
            df_br = round(base ** n * initial_target_dim)
            res = np.abs(df_br - desired_final_dim)
            return res, n

        i_b, i_d0 = self.behavior.n_new_bins, self._init_target_dim

        def _fmin(d0):
            return ndiff(b=i_b, d0=d0)[0]

        bounds = (2, i_b + 1)

        x_best = 1
        y_best = _fmin(x_best)
        for j_d0 in range(bounds[0], bounds[1]):
            if _fmin(j_d0) < y_best:
                x_best = j_d0
                y_best = _fmin(j_d0)

        info(star_string(
            f"Can reach a difference of {y_best} with init target dim  of {x_best} after {ndiff(i_b, x_best)[1]} splits. Adjusting..."))
        return x_best

    def _restart(self, length: float = None) -> None:
        """
        Reset TR observations, reset counter, reset base length

        Args:
            length: new base length after resetting, if not set, length_init will be used.

        """
        self._X = np.empty((0, self.target_dim))
        self._fX = np.empty((0, 1))

        self.failcount = 0
        self.succcount = 0
        if length is None:
            self.length = self.length_init
        else:
            self.length = length

    @property
    def failtol(self) -> float:
        """
        The fail tolerance for the BAxUS algorithm.
        Is computed dynamically depending on the split we are in as the fail tolerance is dependent on the
        current target dimensionality.

        Returns: the fail tolerance for the BAxUS algorithm

        """
        ft_max = np.max([4.0, self.target_dim])
        if self.target_dim == self.input_dim:
            return ft_max

        desired_final_dim = self.input_dim
        evaluation_budget = self.max_evals if self.behavior.budget_until_input_dim == 0 else self.behavior.budget_until_input_dim
        evaluation_budget = evaluation_budget - self._budget_lost_in_previous_trs

        psi = 1
        new_bins_on_split = self.behavior.n_new_bins
        _log_base = psi * new_bins_on_split + 1
        n = round(math.log(desired_final_dim / self._init_dim_in_tr, _log_base))  # splits

        def _budget(dim):

            return (evaluation_budget * dim * (1 - _log_base)) / (self._init_dim_in_tr * (1 - _log_base ** (n + 1)))

        budget = _budget(self.target_dim)

        del (
            psi,
            new_bins_on_split,
            evaluation_budget,
        )

        length_init = self.behavior.initial_base_length

        gamma = 2 * math.log(self.length_min / length_init, 0.5)
        if gamma == 0:
            return ft_max
        ft = math.ceil(budget / gamma)
        failtol = max(1, min(ft, ft_max))

        return failtol

    def _adjust_length(self, fX_next) -> None:
        """
        Adjust the base length of the current trust region depending on the outcome of the next evaluation.
        If the next evaluation is better than the current, increase success count and potentially increase TR base length.
        Otherwise, increase fail count and potentially decrease TR base length.

        Args:
            fX_next: the function value of the next point

        """
        debug(
            f"eval {self.n_evals}: length = {self.length}, failcount = {self.failcount} (failtol = {self.failtol}), "
            f"succcount = {self.succcount} (succtol = {self.succtol})"
        )
        prev_data = self._fX

        if np.min(fX_next) < np.min(
                prev_data
        ) - self.behavior.success_decision_factor * math.fabs(np.min(prev_data)):
            debug(f"eval {self.n_evals}: increase success count")
            self.succcount += 1
            self.failcount = 0
        else:
            debug(f"eval {self.n_evals}: increase failure count")
            self.succcount = 0
            self.failcount += 1
        if self.succcount == self.succtol:  # Expand trust region
            debug(f"eval {self.n_evals}: expanding trust region")
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            debug(f"eval {self.n_evals}: shrinking trust region")
            self.length /= 2.0
            self.failcount = 0

        self._log_property("length_history", f"{self.n_evals}:{self.length}")

    def _choose_splitting_dim(
            self,
            projector: AxUS,
    ) -> Dict[int, int]:
        """
        Choose a new splitting dim based on our defined behavior

        Args:
            projector: the projection model used

        Returns: the new splitting dim or -1 if none could be found


        """

        n_dims_to_split = self.target_dim

        n_new_bins = self.behavior.n_new_bins
        n_new_bins = (n_new_bins + 1) * n_dims_to_split
        assert n_new_bins >= 2 * n_dims_to_split, (
            "Number of new bins has "
            "to be at least 2 times"
            "the number of dimensions"
            "to split"
        )
        weights = self._dimension_importances
        indices_with_lengthscales = {i: weights[i] for i in range(self.target_dim)}
        indices_sorted_by_lengthscales = sorted(
            [i for i in indices_with_lengthscales.keys()],
            key=lambda i: indices_with_lengthscales[i],
        )
        splittable_idxs = np.array(
            [
                i
                for i in indices_sorted_by_lengthscales
                if len(projector.contributing_dimensions(i)) > 1
            ]
        )
        n_dims_to_split = min(len(splittable_idxs), n_dims_to_split)
        if n_dims_to_split == 0:
            return {}
        n_bins_per_dim = n_new_bins // n_dims_to_split
        bins_per_dim = np.array(
            [
                min(n_bins_per_dim, len(projector.contributing_dimensions(i)))
                for i in splittable_idxs
            ]
        )
        cum_sum = np.cumsum(bins_per_dim)
        dims_to_split = np.sum(cum_sum <= n_new_bins)
        dims_and_bins = {
            splittable_idxs[i]: bins_per_dim[i] for i in range(dims_to_split)
        }

        return dims_and_bins

    def _resample_and_restart(self, n_points: int, length: float = None) -> None:
        """
        Resample new initial points and reset algorithm.

        Args:
            n_points: number of new initial points
            length: new base length after resetting

        Returns: None

        """
        # Initialize parameters
        self._restart(length=length)

        # Generate and evaluate initial design points
        n_pts = min(self.max_evals - self.n_evals, n_points)
        X_init = one_around_origin_latin_hypercube(n_pts, self.target_dim)

        X_init_up = from_1_around_origin(
            self.projector.project_up(X_init.T).T, self.lb, self.ub
        )
        fX_init = np.array([[self.f(x)] for x in X_init_up])
        # Update budget and set as initial data for this TR
        self.n_evals += n_pts
        self._X = deepcopy(X_init)
        self._fX = deepcopy(fX_init)

        # Append data to the global history
        self.X = np.vstack((self.X, deepcopy(X_init_up)))
        self.fX = np.vstack((self.fX, deepcopy(fX_init)))

        self._data_dims.extend([self.target_dim] * n_pts)

    @staticmethod
    def _projector_as_base64(projector: ProjectionModel) -> str:
        """
        Return the current projection model as a Base64 string.
        Args:
            projector: the projector to return as base64.

        Returns: the current projection model as a Base64 string.

        """
        if isinstance(projector, AxUS):
            return base64.b64encode(pickle.dumps(projector)).decode("utf-8")
        return ""

    def optimize(self) -> None:
        """
        Run the optimization

        Returns: None

        """
        self._log_property(
            "projectors", f"{self.n_evals}:{self._projector_as_base64(self.projector)}"
        )

        while self.n_evals < self.max_evals and not self._optimum_reached():
            n_pts = min(self.max_evals - self.n_evals, self.n_init)
            # only executed if we already gathered data, i.e., not in the first run
            if len(self._fX) > 1:
                # target dim increase
                n_evals, fbest = self.n_evals, self._fX.min()
                info(f"{n_evals}) Restarting with fbest = {fbest:.4}")

                # Split target dimension, will be used if we made progress and if not -1
                dims_and_bins = self._choose_splitting_dim(self.projector)
                # first_split = self.target_dim == self._init_target_dim  # TODO remove

                if dims_and_bins:  # if we have a remaining-splitting dim
                    splitting_dims = list(dims_and_bins.keys())
                    n_new_bins = sum(list(dims_and_bins.values()))
                    self._log_property(
                        "splitting_dims",
                        f"{self.target_dim_increases}:{','.join([str(x) for x in splitting_dims])}",
                    )
                    self._log_property("split_points", f"{self.n_evals}")
                    self._split_points.append(self.n_evals)
                    for splitting_dim, n_bins in dims_and_bins.items():
                        info(
                            f"eval {self.n_evals}: splitting dimension {splitting_dim + 1} into {n_bins} new "
                            f"bins with lengthscale: {self.lengthscales[splitting_dim]:.4} and contributing input "
                            f"dimensions {sorted(self.projector.contributing_dimensions(splitting_dim))}"
                        )
                    self.projector.increase_target_dimensionality(dims_and_bins)
                    # self.projector.merge_dims(*np.argsort(-dim_ent)[:2])
                    self._log_property(
                        "projectors",
                        f"{self.n_evals}:{self._projector_as_base64(self.projector)}",
                    )
                    self.target_dim += n_new_bins - len(dims_and_bins)
                    self._dim_in_iterations[self.n_evals] = self.target_dim
                    info(
                        f"eval {self.n_evals}: new target dim = {self.target_dim}"
                    )
                    self._axus_change_iterations.append(self.n_evals)
                    self.length = self.behavior.initial_base_length

                    self._X = join_data(self._X, dims_and_bins)

                else:
                    warning(
                        f"eval {self.n_evals}: cannot increase further. "
                        f"Re-starting with new HeSBO embedding and new TR."
                    )
                    self._log_property("tr_die_outs", f"{self.n_evals}")
                    self.projector = AxUS(
                        input_dim=self._input_dim,
                        target_dim=self.target_dim,
                        bin_sizing=self.behavior.embedding_type,
                    )
                    self._log_property(
                        "projectors",
                        f"{self.n_evals}:{self._projector_as_base64(self.projector)}",
                    )
                    self._resample_and_restart(
                        n_points=self.n_init, length=self.length_init
                    )
                    self._axus_change_iterations.append(self.n_evals)
                    self._trust_region_restarts.append(self.n_evals)
                    self._dim_in_iterations[self.n_evals] = self.target_dim

                self.failcount = 0
                self.succcount = 0
            else:
                self._resample_and_restart(self.n_init, self.length_init)
                fbest = self._fX.min()
                info(f"eval {self.n_evals}: starting from fbest = {fbest:.4}")

            # Thompson sample to get next suggestions

            while (
                    self.n_evals < self.max_evals
                    and self.length >= self.length_min
                    and not self._optimum_reached()
            ):
                X_next, X_next_up, fX_next = self._inner_optimization_step()
                self._data_dims.extend([self.target_dim] * len(X_next))
        self._optimized = True
        self._log_property("final_target_dim", self.target_dim)
