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

import math
from typing import Tuple, Dict, Any, List, Callable

import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    MaternKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from torch import Tensor

from baxus.util.behaviors.gp_configuration import MLLEstimation, GPBehaviour
from baxus.util.gp_utils import (
    initializer_factory,
    mle_optimization,
    latin_hypercube_hp_grid,
    pick_best_from_configurations,
)


class GP(SingleTaskGP):
    """
    Extension of a single class GP for our purposes.

    Args:
        train_x: the x-values of the training points
        train_y: the function values of the training points
        likelihood: the likelihood to use
        ard_dims: the number of ARD dimensions
        lengthscale_constraint: the constraints for the lengthscales
        outputscale_constraint: the constraints for the signal variances
    """

    def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            ard_dims,
            lengthscale_constraint=None,
            outputscale_constraint=None,
    ):
        super(GP, self).__init__(
            train_x,
            torch.unsqueeze(train_y, 1) if train_y.ndim == 1 else train_y,
            likelihood,
        )
        self.likelihood = likelihood
        self.mean_module = ConstantMean()

        base_kernel = MaternKernel(
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=ard_dims,
            nu=2.5,
        )
        self.covar_module = ScaleKernel(
            base_kernel, outputscale_constraint=outputscale_constraint
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Call the GP

        Args:
            x: points

        Returns: MultivariateNormal distribution

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @property
    def lengthscales(self) -> np.ndarray:
        """
        return the lengthscales of the base kernel depending on the kernel type
        """
        weights = (
            self.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        )
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(
            np.power(weights, 1.0 / len(weights))
        )  # We now have weights.prod() = 1
        return weights


def train_gp(
        train_x: Tensor,
        train_y: Tensor,
        use_ard: bool,
        gp_behaviour: GPBehaviour = GPBehaviour(),
        hypers=None,
) -> Tuple[GP, Dict[str, Any]]:
    """
    Fit a GP where train_x is in [-1, 1]^D

    Args:
        train_x: training data
        train_y: training data
        use_ard: whether to use automatic relevance detection kernel
        gp_behaviour: the configuration of the GP
        hypers: hyperparameters for the GP, if passed, the GP won't be re-trained

    Returns:

    """
    if hypers is None:
        res_hypers = {}
    else:
        res_hypers = hypers
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 10.0)
    else:
        lengthscale_constraint = Interval(
            0.005, math.sqrt(train_x.shape[1])
        )  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)  # TODO
    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
        device=train_x.device, dtype=train_x.dtype
    )
    ard_dims = (
        (train_x.shape[1]) if use_ard else None
    )
    model = GP(
        train_x=train_x,
        train_y=train_y,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        likelihood=likelihood,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Set model to training mode
    model.train()
    likelihood.train()

    # Initialize an empty hyperparamter "grid" for multistart GD
    hyperparameter_grid = {}
    # Initialize an empty list of model initializers, used later in multi-start GD
    model_initializers: List[Callable[[GP], None]] = []

    # If we passed an existing hyperparameter configuration for this model, use it
    if res_hypers:
        hyperparameter_config = lambda m: m.load_state_dict(res_hypers)
        model_initializers.append(hyperparameter_config)
        model.load_state_dict(res_hypers)
    else:
        # Otherwise add bounds and default values to hyperparameter grid
        hyperparameter_grid["covar_module.outputscale"] = (0.05, 20.0, 1.0)

        hyperparameter_grid["covar_module.base_kernel.lengthscale"] = (
            0.005,
            10.0,
            0.5,
        )
        hyperparameter_grid["likelihood.noise"] = (5e-4, 0.2, 0.005)
        samples = latin_hypercube_hp_grid(
            hyperparameter_grid, gp_behaviour.n_initial_samples
        )

        # convert hyperparameter priors to initializers
        for i in range(gp_behaviour.n_initial_samples):
            hyperparameter_config = {}
            for k, v in samples.items():
                hyperparameter_config[k] = v[i]

            initializer = initializer_factory(hyperparameter_config)
            model_initializers.append(initializer)
        if gp_behaviour.mll_estimation == MLLEstimation.LHS_PICK_BEST_START_GD:
            model_initializers = pick_best_from_configurations(
                initializers=model_initializers,
                model=model,
                train_x=train_x,
                train_y=train_y,
                n_best=gp_behaviour.n_best_on_lhs_selection,
            )

    # save the state dicts for multi-start gradient descent
    best_loss = np.inf
    best_state_dict = None
    for i, initializer in enumerate(model_initializers):
        state_dict, loss = mle_optimization(
            initializer=initializer,
            model=model,
            num_steps=gp_behaviour.n_mle_training_steps,
            train_x=train_x,
            train_y=train_y,
        )
        if loss < best_loss:
            best_state_dict = state_dict
            best_loss = loss
        else:
            del state_dict

    model.load_state_dict(best_state_dict)
    res_hypers = best_state_dict

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model, res_hypers
