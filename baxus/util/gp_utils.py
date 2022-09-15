from copy import deepcopy
from typing import Dict, Callable, Tuple, List, Optional, OrderedDict

import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from scipy.stats import qmc
from torch import Tensor

from baxus.util.utils import from_unit_cube


def pick_best_from_configurations(
        initializers: List[Callable[["baxus.gp.GP"], None]],
        model: "baxus.gp.GP",
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        n_best: Optional[int] = 1,
) -> List[Callable[["baxus.gp.GP"], None]]:
    """
    Pick the n_best best performing initializers from a list of initializers based on a GP and a MLL
    Args:
        initializers: list of initializers, sets GP hyperparameters
        model: the GP model
        train_x: the data to evaluate the model likelihood on
        train_y: the data to evaluate the model likelihood on
        n_best: number of best performing initializers to choose

    Returns: list of initializer functions

    """
    assert n_best <= len(initializers), "At most as many best as we have initializers"
    # avoid side effects
    model = deepcopy(model)

    losses = []
    for i, initializer in enumerate(initializers):
        initializer(model)
        model.train()
        model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        output = model(train_x)
        loss = -mll(output, train_y).cpu().detach().numpy()
        losses.append(loss)
    return np.array(initializers)[np.argsort(losses)[:n_best]].tolist()


def mle_optimization(
        initializer: Callable[["baxus.gp.GP"], None],
        model: "baxus.gp.GP",
        num_steps: int,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
) -> Tuple[OrderedDict[str, Tensor], float]:
    """
    Optimize likelihood of a model with an initializer.
    :param initializer: the model initializer
    :param model: the GP model
    :param num_steps: number gradient descent steps
    :param kernel_type: the kernel type of the GP model
    :param train_x: the training data
    :param train_y: the training data
    :param mll: the model likelihood
    :return: state dict and the average loss
    """
    # avoid side effects
    model = deepcopy(model)
    initializer(model)
    model.train()
    model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)  # TODO

    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    # only use half of the optimizer steps if kplsk kernel
    cum_loss = 0
    for _ in range(
            num_steps
    ):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        cum_loss += loss
        loss.backward()
        optimizer.step()
    return deepcopy(model.state_dict()), cum_loss / num_steps if num_steps > 0 else 0


def initializer_factory(
        hyperparameter_configuration: Dict[str, float]
) -> Callable[["turbo.gp.GP"], None]:
    """
    Take a hyperparameter configuration and return a lambda initializing a model with this configuration
    :param hyperparameter_configuration: the hyperparameter configuration
    :return: callabe, defined in GPyTorch model
    """
    return lambda m: m.initialize(**hyperparameter_configuration)


def latin_hypercube_hp_grid(
        hyperparameter_grid: Dict[str, Tuple[float, float, float]], n_samples: int
) -> Dict[str, np.ndarray]:
    """
    Draw samples from latin hypercube from hyperparameter grid. Default configuration will always be the first configuration.
    :param hyperparameter_grid: dictionary, key: hyperparameter name, value: Tuple[lower_bound, upper_bound, default value]
    :param n_samples: number of samples to return, if 1 return default values
    :return: dictionary, key: hyperparameter name, value: np.ndarray of sample values (shape: (n_samples, 1))
    """
    return_configs = {}
    for k, v in hyperparameter_grid.items():
        return_configs[k] = np.array([v[2]])
        # if only one sample, return the default value
    if n_samples == 1:
        return return_configs
    hp_grid = deepcopy(hyperparameter_grid)
    keys = []
    bounds = np.empty((0, 2))
    for k, v in hp_grid.items():
        bounds = np.vstack((bounds, v[:2]))
        keys.append(k)
    d = len(keys)
    sampler = qmc.LatinHypercube(d=d)
    sample = sampler.random(n=n_samples - 1)
    samples = from_unit_cube(sample, bounds[:, 0], bounds[:, 1])
    for i, k in enumerate(keys):
        return_configs[k] = np.hstack((return_configs[k], samples[:, i]))
    return return_configs
