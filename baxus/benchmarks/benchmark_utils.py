import os
from copy import deepcopy, copy
from logging import warning, info
from typing import List

import numpy as np
import pandas as pd


MAX_RETRIES = 1


def run_and_plot(
        m: "baxus.benchmarks.OptimizationMethod",
        repetitions: List[int],
        directory: str,
) -> None:
    """
    Run an experiment for a certain number of repetitions and save the results

    Args:
        m: the experiment to runm_x
        repetitions: the repetitions to run
        directory: the directory to save the results

    Returns:
        None
    """
    os.makedirs(directory, exist_ok=True)

    base_run_dir = copy(m.run_dir)

    for rep in repetitions:
        out_path = os.path.join(directory, f"repet_{rep}.csv.xz")
        rep_run_dir = os.path.join(base_run_dir, f"repetition_{rep}")
        os.makedirs(rep_run_dir, exist_ok=True)
        m.run_dir = rep_run_dir

        if os.path.exists(out_path):
            continue
        info(f"starting repetition {rep}")
        for mr in range(MAX_RETRIES):
            try:
                _m = deepcopy(m)
                _m.reset()
                _m.optimize()
                break
            except Exception as e:
                if mr == MAX_RETRIES - 1:
                    raise e
                warning(f"Optimization failed. Retrying... ({mr + 1}/{MAX_RETRIES})")
        m_x, m_y_raw = _m.optimization_results_raw()
        _, m_y = _m.optimization_results_incumbent()
        m_y_raw = np.expand_dims(m_y_raw, axis=1)
        if m_x is not None:
            columns = [f"x{i}" for i in range(m_x.shape[1])] + ["y_raw"]
            r_df = pd.DataFrame(np.concatenate((m_x, m_y_raw), axis=1), columns=columns)
        else:
            columns = [["y_raw"]]
            r_df = pd.DataFrame(m_y_raw, columns=columns)
        r_df.to_csv(out_path)
        del r_df
        del m_y_raw
