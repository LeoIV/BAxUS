import math
from typing import Union

import gpytorch
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement as _EI
from botorch.optim import optimize_acqf

from baxus.gp import GP


class ExpectedImprovement:
    def __init__(self, gp: GP, best_f: Union[float, np.ndarray], lb: np.ndarray, ub: np.ndarray,
                 evaluation_batch_size: int = 100, ):
        self.ub = ub
        self.lb = lb
        self.evaluation_batch_size = evaluation_batch_size
        self.best_f = best_f
        self.gp = gp
        self._EI = _EI(model=self.gp, best_f=self.best_f)

    def __call__(self, X: np.ndarray):

        def _ei(X):
            X = np.expand_dims(X, 1)
            return torch.unsqueeze(self._EI(torch.unsqueeze(torch.tensor(X), 1)), 1).detach().numpy()

        if X.ndim == 1:
            X = X[np.newaxis, :]
        if len(X) > 100:
            # batched version
            Xs = np.split(X, math.ceil(len(X) / self.evaluation_batch_size))
            eis = [_ei(_X) for _X in Xs]
            result = np.concatenate(eis)
        else:
            result = _ei(X)
        return result

    def optimize(self):
        with gpytorch.settings.max_cholesky_size(2000):
            X_cand, y_cand = optimize_acqf(
                acq_function=self._EI,
                bounds=torch.tensor([self.lb.reshape(-1), self.ub.reshape(-1)]),
                q=1,
                num_restarts=20,
                raw_samples=100,
                options={},
            )
        return X_cand.detach().numpy(), y_cand.detach().numpy()
