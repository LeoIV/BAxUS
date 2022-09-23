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
# Author: anonymous

import lzma
import math
import os
import sys
import time
from copy import deepcopy
from logging import info, debug, warning
from typing import Tuple, Optional, Any, Union, Dict
from zipfile import ZipFile, ZIP_LZMA

import gpytorch
import numpy as np
import torch

from baxus.benchmarks.benchmark_function import Benchmark
from baxus.benchmarks.other_methods import OptimizationMethod
from baxus.gp import train_gp
from baxus.util.acquisition_function_types import AcquisitionFunctionType
from baxus.util.acquisition_functions import ExpectedImprovement
from baxus.util.behaviors import EmbeddedTuRBOBehavior
from baxus.util.behaviors.gp_configuration import GPBehaviour
from baxus.util.projections import AxUS
from baxus.util.space_learning.trust_region import create_Xcand
from baxus.util.utils import (
    one_around_origin_latin_hypercube,
    from_1_around_origin,
)


class EmbeddedTuRBO(OptimizationMethod):
    """
        Embedded TuRBO is the base class for BAxUS. It is the implementation used for our ablation studies and runs
        TuRBO in an embedded space.

        Args:
            f: the benchmark function
            target_dim: the target dimensionality
            n_init: the number of initial samples
            max_evals: the maximum number of evaluations
            behavior: the behavior configuration of the algorithm
            gp_behaviour: the behavior of the GP
            verbose: whether to print verbose log messages
            use_ard: whether to use an ARD kernel
            max_cholesky_size: If the size of a LazyTensor is less than max_cholesky_size, then root_decomposition and inv_matmul of LazyTensor will use Cholesky rather than Lanczos/CG.
            dtype: the data type to use
            run_dir: the directory to write run information to
            conf_name: the name of the current configuration
        """

    def __init__(
            self,
            f: Benchmark,
            target_dim: int,
            n_init: int,
            max_evals: int,
            behavior: EmbeddedTuRBOBehavior = EmbeddedTuRBOBehavior(),
            gp_behaviour: GPBehaviour = GPBehaviour(),
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            dtype="float64",
            run_dir: str = ".",
            conf_name: Optional[str] = None,
    ):

        self.behavior = behavior
        super().__init__(conf_name=conf_name, run_dir=run_dir)
        # Very basic input checks

        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0

        assert gp_behaviour.n_mle_training_steps >= 30 and isinstance(
            gp_behaviour.n_mle_training_steps, int
        )
        assert max_evals > n_init
        assert dtype == "float32" or dtype == "float64"
        if target_dim > f.dim:
            warning(
                f"Target dimension {target_dim} is larger than the input dimension {f.dim}. Setting target dimension to input dimension for function {type(f).__name__}."
            )
            target_dim = f.dim

        # Save function information
        self._target_dim = target_dim
        self._input_dim = f.dim
        self.f = f
        self.lb = f.lb_vec
        self.ub = f.ub_vec

        # Settings

        self.n_init = n_init
        self.max_evals = max_evals
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.gp_behaviour = gp_behaviour
        self.n_evals = 0
        if self._input_dim != self._target_dim:
            info(f"eval {self.n_evals}: creating HeSBO embedding for TuRBO instance...")
            self.projector = AxUS(
                self._input_dim,
                self._target_dim,
                bin_sizing=self.behavior.embedding_type,
            )
            try:
                eff_dims = self.f.effective_dims
                info(
                    f"important target dims: {sorted(list(set([self.projector.input_to_target_dim[d] for d in eff_dims])))}"
                )
            except:
                pass
        else:
            self.projector = False

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = (
            np.zeros((0, self._target_dim)) if self.use_ard else np.zeros((0, 1))
        )

        # Tolerances and counters
        self.succtol = self.behavior.success_tolerance

        # Trust region sizes
        self._length_min = self.behavior.min_base_length
        self._length_max = self.behavior.max_base_length
        self._length_init = self.behavior.initial_base_length

        # Save the full history
        self.X = np.zeros((0, self._input_dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

        # Initialize parameters
        self._restart()

        # History
        self._fds = {}
        self._model_history_archive = "model_history.zip"

        info(f"Running with the following behavior\n\n{self.behavior.pretty_print()}")

    @property
    def failtol(self) -> float:
        """
        The fail tolerance of the current trust region.

        Returns: the fail tolerance (=max(4, current target dimensionality))

        """
        failtol = np.ceil(
            np.max(
                [
                    4.0,
                    self._target_dim
                ]
            )
        )
        return failtol

    @property
    def conf_dict(self) -> Dict[str, Any]:
        """
        The current behavior configuration as a dictionary

        Returns: the current behavior configuration as a dictionary

        """
        return {**super().conf_dict, **self.behavior.conf_dict}

    @property
    def n_cand(self) -> int:
        """
        The number of candidates for the discrete Thompson sampling

        Returns: the number of candidates for the discrete Thompson sampling

        """
        return min(100 * self._target_dim, 5000)

    @property
    def target_dim(self) -> int:
        """
        The target dimensionality.

        Returns: the target dimensionality

        """
        return self._target_dim

    @target_dim.setter
    def target_dim(self, target_dim: int) -> None:
        """
        Setter for the target dimensionality

        Args:
            target_dim:  the new target dimensionality

        Returns:

        """
        self._target_dim = target_dim

    @property
    def input_dim(self) -> int:
        """
        The input dimensionality

        Returns: the input dimensionality

        """
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim: int):
        """
        Setter for the input dimensionality.

        .. warning::
            Should not be called, throws an error when called.

        Args:
            input_dim: the new input dimensionality

        Returns:

        """
        raise AttributeError("Cannot change input dim")

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

    def reset(self) -> None:
        """
        Reset the state of the current instance (re-initiate the projector, reset global observations, reset local
        observations, reset fail- and success counts). Does not reset the target dimensionality

        Returns: None

        """
        self.projector = AxUS(
            self._input_dim, self._target_dim, bin_sizing=self.behavior.embedding_type
        )
        self.X = np.zeros((0, self._input_dim))
        self.fX = np.zeros((0, 1))
        self.length = self.length_init
        if hasattr(self.f, "effective_dims") and isinstance(self.f.effective_dims, np.ndarray):
            self._log_property("function_effective_dims", self.f.effective_dims)
        self._restart()

    def _resample_and_restart(self, n_points: int, length: float = None) -> None:
        """
        Resample new initial points and reset algorithm

        Args:
            n_points: number of new points to sample
            length: new trust region base length after reset

        Returns: None

        """
        # Initialize parameters
        self._restart(length=length)

        # Generate and evaluate initial design points
        n_pts = min(self.max_evals - self.n_evals, n_points)
        X_init = one_around_origin_latin_hypercube(n_pts, self._target_dim)

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

    def _restart(self, length: Optional[float] = None) -> None:
        """
        Reset observations, reset counters, reset trust region base length

        Args:
            length:  new trust region base length after resetting

        Returns: None

        """
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        if length is None:
            self.length = self.length_init
        else:
            self.length = length

    def _adjust_length(self, fX_next: np.ndarray) -> None:
        """
        Adjust the base length of the current trust region depending on the outcome of the next evaluation.
        If the next evaluation is better than the current, increase success count and potentially increase TR base length.
        Otherwise, increase fail count and potentially decrease TR base length.

        Args:
            fX_next: the function value of the next point

        """
        debug(
            f"eval {self.n_evals}: failcount = {self.failcount} (failtol = {self.failtol}), "
            f"succcount = {self.succcount} (succtol = {self.succtol})"
        )
        if np.min(fX_next) < np.min(
                self._fX
        ) - self.behavior.success_decision_factor * math.fabs(np.min(self._fX)):
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

    def _create_candidates(
            self,
            X: np.ndarray,
            fX: np.ndarray,
            length: float,
            gp_behaviour: GPBehaviour,
            hypers,
            tr_idx: Optional[int] = None,
            multiple_lengthscales: bool = False,
    ) -> Optional[
        Union[
            Tuple[
                Tuple,
                Dict[str, Any],
                np.ndarray,
                np.ndarray,
            ],
            Tuple[
                Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]
            ],
        ]
    ]:
        """
        Generate candidates assuming X has been scaled to [-1,1]^d.

        Args:
            X: the local TR data x-values
            fX: the local TR data y-values
            global_X: the global x-values (used for fitting a PLS if required)
            global_y: the global y-values (used for fitting a PLS if required)
            length: the current base length
            gp_behaviour: the behavior definition of the GP
            hypers: the pre-computed GP hyperparameters. If empty, the GP will be trained anew
            tr_idx: the trust region index (for TuRBO-m)
            multiple_lengthscales: whether to use multiple lengthscales
            use_pls: whether to use a PLS kernel
            n_pls_components: number of PLS components for PLS kernel
            kernel_type: the kernel type of the PLS kernel (only recognized if use_pls is true)
            pls: pre-computed PLS. If not given, a new PLS is computed
            turbo_1_return_format: whether to use the TuRBO-1 return format (supports multiple acquisition functions)
        Returns:
            either a tuple (X_candidates, y_candidates, dict of GP hyperparams, PLSContainer, lb of TR, ub of TR) <- TheSBO-1 return format or (dict of best per acquisition function, dict of GP hyperparameters, PLSContainer)
        """
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        target_dim = self._target_dim if tr_idx is None else self.target_dims[tr_idx]

        fX = fX.copy() * (-1)
        # Standardize local function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        device, dtype = torch.device("cpu"), self.dtype
        len_hypers = len(hypers)  # save here as overwritten later
        # We use CG + Lanczos for training if we have enough data

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)

            # pass stored pls unless we want to retrain
            # Possibly get PLSContainer from GP. If we passed a PLSContainer to train_gp, this is the same one we passed.
            # Otherwise, if we passed None and the kernel requires a PLS, it will be a newly trained PLS.
            gp, hyper = train_gp(
                train_x=X_torch,
                train_y=y_torch,
                use_ard=self.use_ard,
                gp_behaviour=gp_behaviour,
                hypers=hypers,
            )
        if self.n_evals % 10 == 0 and len_hypers == 0:
            # save model
            full_arch_path = os.path.join(self.run_dir, self._model_history_archive)
            with ZipFile(
                    full_arch_path,
                    "a" if os.path.exists(full_arch_path) else "w",
                    compression=ZIP_LZMA,
            ) as zip_archive:
                model_path = (
                    f"gp_iter_{self.n_evals}.pth"
                    if tr_idx is None
                    else f"gp_iter_{self.n_evals}_tr_{tr_idx}.pth"
                )
                with zip_archive.open(model_path, "w") as comp_f:
                    torch.save(gp, comp_f)

        # Create the trust region boundaries
        x_center = X[fX.argmax().item(), :][None, :]
        # x_center = gp_X[gp_y.argmin().item(), :][None, :]
        self._log_property(
            "tr_centers" if not multiple_lengthscales else f"tr_{tr_idx}_centers",
            f"{self.n_evals}:{x_center.tolist()}",
        )
        weights = gp.lengthscales
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(
            np.power(weights, 1.0 / len(weights))
        )

        if not multiple_lengthscales:
            self.lengthscales = weights
        else:
            self.lengthscales[tr_idx] = weights
        self._log_property(
            "lengthscales"
            if not multiple_lengthscales
            else f"lengthscales_tr_{tr_idx}",
            f"{self.n_evals}:{weights.tolist()}",
        )
        X_cand, lb, ub = create_Xcand(
            x_center=x_center,
            weights=weights,
            length=length,
            dim=target_dim,
            n_cand=self.n_cand,
            dtype=dtype,
            device=device,
        )

        if X_cand.size == 0:
            return None

        # Figure out what device we are running on
        device, dtype = torch.device("cpu"), self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        best_per_acq = None

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad() if AcquisitionFunctionType.EXPECTED_IMPROVEMENT != self.behavior.acquisition_function else gpytorch.settings.max_cholesky_size(
                self.max_cholesky_size):

            if self.behavior.acquisition_function == AcquisitionFunctionType.THOMPSON_SAMPLING:
                X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
                y_cand = (
                    gp.likelihood(gp(X_cand_torch))
                        .sample(torch.Size([1]))
                        .t()
                        .cpu()
                        .detach()
                        .numpy()
                )
                best_per_acq = (X_cand, (mu + sigma * y_cand) * (-1))
                del X_cand_torch
            elif self.behavior.acquisition_function == AcquisitionFunctionType.EXPECTED_IMPROVEMENT:
                EI = ExpectedImprovement(gp, best_f=fX.max(), lb=lb, ub=ub)
                start = time.time()
                X_cand, y_cand = EI.optimize()
                end = time.time()
                debug(
                    f"Optimizing EI took {end - start:.2f}s in {self.target_dim} dims with {len(self._X)} datapoints.")
                del EI
                # y_cand = torch.unsqueeze(y_cand, 1)

                best_per_acq = (
                    X_cand,
                    (mu + sigma * y_cand) * (-1),
                )

        # Remove the torch variables
        del X_torch, y_torch, gp
        return best_per_acq, hypers, lb, ub

    def _log_property(self, property_name: str, value: Any) -> None:
        """
        Log a property to a file. If the file descriptor does not already exist, it is created, otherwise an
        already opened file descriptor is used.

        Args:
            property_name: the property to log. This will determine the file name
            value: the value to log. This is just appended to the file if it already exists.

        Returns: None

        """
        path = os.path.join(self.run_dir, f"{property_name}.txt.xz")
        if property_name not in self._fds:
            self._fds[property_name] = lzma.open(path, "wt")
        self._fds[property_name].write(f"{value}\n")

    def _select_candidates(self, best_per_acq: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Choose the next evaluation point.

        Args:
            best_per_acq: Tuple of x-values and acquisition function values of the candidates.

        Returns: The next point according to the acquisition function selected.

        """
        """Select candidates."""
        X_next = np.ones((1, self._target_dim))
        indbests = []
        X_cand, y_cand = best_per_acq
        # Pick the best point and make sure we never pick it again
        if self.behavior.acquisition_function == AcquisitionFunctionType.THOMPSON_SAMPLING:
            indbest = np.argmin(y_cand[:, 0])
        elif self.behavior.acquisition_function == AcquisitionFunctionType.EXPECTED_IMPROVEMENT:
            if y_cand.size > 1:
                indbest = np.argmax(y_cand[:, 0])
            else:
                indbest = 0
        else:
            raise RuntimeError("unknown acquisition function type")
        indbests.append(indbest)
        X_next[0, :] = deepcopy(X_cand[indbest, :])
        del X_cand, y_cand

        return X_next, np.array(indbests).squeeze()

    def _inner_optimization_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create candidates, select candidate, project up point, evaluate point

        Returns: next point in target space, next point in input space, function value of the next point

        """
        # Warp inputs
        X = self._X
        fX = deepcopy(self._fX).ravel()

        # Create th next batch
        is_cands = self._create_candidates(
            X,
            fX,
            length=self.length,
            gp_behaviour=self.gp_behaviour,
            hypers={},
        )
        best_per_acq, hypers, lb, ub = is_cands

        # select next batch
        X_next, _ = self._select_candidates(best_per_acq)

        # Undo the warping
        X_next_up = from_1_around_origin(
            self.projector.project_up(X_next.T).T if self.projector else X_next,
            self.lb,
            self.ub,
        )

        # Evaluate batch
        fX_next = np.array([[self.f(x)] for x in X_next_up])

        # Update trust region
        self._adjust_length(fX_next)

        # Update budget and append data
        self.n_evals += 1
        self._X = np.vstack((self._X, X_next))
        self._fX = np.vstack((self._fX, fX_next))

        debug(
            f"eval {self.n_evals} on {self.f.fun_name}: new point: {fX_next.min():.4} (current global / local best: {self.fX.min():.4}/{self._fX.min():.4})"
        )
        if fX_next.min() < self.fX.min():
            n_evals, fbest = self.n_evals, fX_next.min()
            info(f"eval {self.n_evals} on {self.f.fun_name}: new best: {fbest:.4}")

        # Append data to the global history
        self.X = np.vstack((self.X, deepcopy(X_next_up)))
        self.fX = np.vstack((self.fX, deepcopy(fX_next)))

        return X_next, X_next_up, fX_next

    def _optimum_reached(self, tolerance: float = 1e-3) -> bool:
        """
        Whether the optimum was reached according to some absolute tolerance value

        Args:
            tolerance: the absolute tolerance. If the difference of the best function value to the optimal function value is less than this, return true. False otherwise.

        Returns: True, if the difference of the best function value to the optimal function value is less than the tolerance. False otherwise.

        """
        try:
            optimum = np.array(self.f.optimal_value).squeeze()
            current_best = np.min(self.fX)
            optimum_reached = math.isclose(optimum, current_best, abs_tol=tolerance)
            if optimum_reached:
                info(
                    f"Optimum reached within a tolerance of {tolerance}. Stopping early..."
                )
            return optimum_reached
        except:
            return False

    def optimize(self) -> None:
        """
        Run the optimization until the maximal number of evaluations or the optimum are reached.

        Returns: None

        """

        while self.n_evals < self.max_evals and not (self.behavior.noise > 0 or self._optimum_reached()):
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                info(f"eval {self.n_evals}: restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evaluate initial design points
            n_pts = min(self.max_evals - self.n_evals, self.n_init)
            X_init = one_around_origin_latin_hypercube(n_pts, self._target_dim)
            X_init_up = from_1_around_origin(
                self.projector.project_up(X_init.T).T if self.projector else X_init,
                self.lb,
                self.ub,
            )
            fX_init = np.array([[self.f(x)] for x in X_init_up])

            # Update budget and set as initial data for this TR
            self.n_evals += n_pts
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init_up)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            fbest = self._fX.min()
            info(f"eval {self.n_evals}: starting from fbest = {fbest:.4}")

            # Thompson sample to get next suggestions
            while (
                    self.n_evals < self.max_evals
                    and self.length >= self.length_min
                    and not self._optimum_reached()
            ):
                self._inner_optimization_step()

        self._optimized = True
        self._log_property("final_target_dim", self.target_dim)

    def _close_fds(self) -> None:
        """
        Close any open file handles.

        Returns: None

        """
        for k, v in self._fds.items():
            info(f"Closing file descriptor for '{k}' logger")
            v.close()
        del self._fds
        self._fds = {}

    def __del__(self):
        """
        Close any open file handles.

        Returns: None

        """
        self._close_fds()

    def optimization_results_raw(
            self,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        The observations in the input space and their function values.

        Returns: The observations in the input space and their function values.

        """
        assert self._optimized, "Model hasn't been optimized yet"
        return self.X, self.fX.squeeze()
