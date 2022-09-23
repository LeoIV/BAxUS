import json
import logging
import os
import sys
from datetime import datetime
from logging import info, warning
from typing import List
from zlib import adler32

from baxus import EmbeddedTuRBO, BAxUS
from baxus.benchmarks import run_and_plot, EffectiveDimBenchmark, EffectiveDimBoTorchBenchmark, RandomSearch
from baxus.util.behaviors import BaxusBehavior, EmbeddedTuRBOBehavior
from baxus.util.behaviors.gp_configuration import GPBehaviour
from baxus.util.exceptions import ArgumentError
from baxus.util.parsing import parse, embedding_type_mapper, acquisition_function_mapper, mle_optimization_mapper, \
    fun_mapper
from baxus.util.utils import star_string

FORMAT = "%(asctime)s %(levelname)s: %(filename)s: %(message)s"
DATEFORMAT = '%m/%d/%Y %I:%M:%S %p'


def main(argstring: List[str]) -> None:
    """
    Parse the argstring and run algorithms based on the definition.

    .. note::
        This function should not be called directly but is called by benchmark_runner.py in the project root.

    Args:
        argstring: the argument string

    Returns: Nothing

    """
    args = parse(argstring)
    directory = os.path.join(
        args.results_dir,
        f"{datetime.now().strftime('%d_%m_%Y')}{f'-{args.run_description}' if len(args.run_description) > 0 else ''}",
    )
    os.makedirs(directory, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(directory, "logging.log"),
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format=FORMAT,
        force=True,
        datefmt=DATEFORMAT
    )

    sysout_handler = logging.StreamHandler(sys.stdout)
    sysout_handler.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))
    logging.getLogger().addHandler(sysout_handler)

    repetitions = list(range(args.num_repetitions))

    args_dict = vars(args)
    with open(os.path.join(directory, "conf.json"), "w") as f:
        f.write(json.dumps(args_dict))

    bin_sizing_method = embedding_type_mapper[args.embedding_type]

    acquisition_function = acquisition_function_mapper[args.acquisition_function]

    mle_optimization_method = mle_optimization_mapper[args.mle_optimization]

    input_dim = args.input_dim
    target_dim = args.target_dim
    n_init = args.n_init
    max_evals = args.max_evals
    noise_std = args.noise_std
    new_bins_on_split = args.new_bins_on_split
    multistart_samples = args.multistart_samples
    mle_training_steps = args.mle_training_steps
    multistart_after_samples = args.multistart_after_sample
    l_init = args.initial_baselength
    l_min = args.min_baselength
    l_max = args.max_baselength
    adjust_initial_target_dim = args.adjust_initial_target_dimension
    budget_until_input_dim = args.budget_until_input_dim

    combs = {}

    if n_init is None:
        n_init = target_dim + 1
    if args.min_baselength > args.max_baselength:
        raise ArgumentError(
            "Minimum baselength has to be larger than maximum baselength."
        )
    if args.input_dim < args.target_dim:
        raise ArgumentError(
            "Input dimension has to be larger than target dimension."
        )
    if args.noise_std < 0:
        raise ArgumentError("Noise standard deviation has to be positive.")
    if max_evals < budget_until_input_dim:
        raise ArgumentError("budget_until_input_dim has to be <= max_evals.")
    if args.multistart_samples < 1:
        raise ArgumentError("Number of multistart samples has to be >= 1.")
    if args.multistart_after_sample > args.multistart_samples:
        raise ArgumentError(
            f"Number of multistart samples after sampling {args.multistart_after_sample} has to be smaller or equal to the numbers"
            f"of initial multistart samples {args.multistart_samples}."
        )
    if args.multistart_after_sample < 1:
        raise ArgumentError(
            "Number of multistart samples after sampling has to be >= 1."
        )
    if args.mle_training_steps < 0:
        raise ArgumentError("Number of mle training steps has to be >= 0.")
    if new_bins_on_split < 2:
        raise ArgumentError("Number of new bins on split has to be greater than one.")

    funs = {
        k: v(dim=input_dim, noise_std=noise_std)
        for k, v in fun_mapper().items()
        if k == args.function
    }

    c = {
        f"{k}_in_dim_{v.dim}_t_dim{target_dim}_n_init_{n_init}"
        f"{f'_noise_{noise_std}' if noise_std > 0 else ''}": {
            "input_dim": v.dim,
            "target_dim": min(v.dim, target_dim),
            "n_init": n_init,
            "f": v,
            "lb": v.lb_vec,
            "ub": v.ub_vec,
        }
        for k, v in funs.items()
    }

    combs.update(c)

    for i, (k, comb) in enumerate(combs.items()):
        info(f"running combination {k}")
        llb = comb["lb"]
        uub = comb["ub"]
        input_dim = comb["input_dim"]
        target_dim = comb["target_dim"]
        n_init = comb["n_init"]

        f = comb["f"]

        function_dir = os.path.join(directory, k)
        os.makedirs(function_dir, exist_ok=True)

        if "baxus" == args.algorithm:
            # *** BAxUS ***
            info("*** BAxUS***")
            behavior = BaxusBehavior(
                n_new_bins=new_bins_on_split,
                initial_base_length=l_init,
                min_base_length=l_min,
                max_base_length=l_max,
                acquisition_function=acquisition_function,
                embedding_type=bin_sizing_method,
                adjust_initial_target_dim=adjust_initial_target_dim,
                noise=noise_std,
                budget_until_input_dim=budget_until_input_dim
            )
            gp_behaviour = GPBehaviour(
                mll_estimation=mle_optimization_method,
                n_initial_samples=multistart_samples,
                n_best_on_lhs_selection=multistart_after_samples,
                n_mle_training_steps=mle_training_steps,
            )
            conf_name = (
                f"baxus_{behavior}_{gp_behaviour}"
            )
            run_dir = os.path.join(
                function_dir,
                str(adler32(conf_name.encode("utf-8"))),
            )
            baxus = BAxUS(
                f=f,  # Handle to objective function
                n_init=n_init,  # Number of initial bounds from an Latin hypercube design
                max_evals=max_evals,  # Maximum number of evaluations
                target_dim=target_dim,
                run_dir=run_dir,
                conf_name=conf_name,
                behavior=behavior,
                gp_behaviour=gp_behaviour,
            )
            run_and_plot(m=baxus, repetitions=repetitions, directory=run_dir)
            del baxus

        if "embedded_turbo_target_dim" == args.algorithm:
            # *** Embedded TuRBO - Target dim ***
            info("*** Embedded TuRBO - Target Dim ***")
            behavior = EmbeddedTuRBOBehavior(
                initial_base_length=l_init,
                min_base_length=l_min,
                max_base_length=l_max,
                acquisition_function=acquisition_function,
                embedding_type=bin_sizing_method,
                noise=noise_std,
            )
            gp_behaviour = GPBehaviour(
                mll_estimation=mle_optimization_method,
                n_initial_samples=multistart_samples,
                n_best_on_lhs_selection=multistart_after_samples,
                n_mle_training_steps=mle_training_steps,
            )
            conf_name = (
                f"embedded_turbo_target_dim_{behavior}"
                f"_{gp_behaviour}"
            )
            run_dir = os.path.join(
                function_dir,
                str(adler32(conf_name.encode("utf-8"))),
            )
            embedded_turbo = EmbeddedTuRBO(
                f=f,  # Handle to objective function
                target_dim=target_dim,
                n_init=n_init,  # Number of initial bounds from an Latin hypercube design
                max_evals=max_evals,  # Maximum number of evaluations
                run_dir=run_dir,
                conf_name=conf_name,
                behavior=behavior,
                gp_behaviour=gp_behaviour,
            )

            run_and_plot(
                m=embedded_turbo,
                repetitions=repetitions,
                directory=run_dir,
            )
            del embedded_turbo

        if "embedded_turbo_effective_dim" == args.algorithm:
            if issubclass(type(f), EffectiveDimBenchmark) or issubclass(
                    type(f), EffectiveDimBoTorchBenchmark
            ):
                effective_dim = f.effective_dim
            else:
                warning("Benchmark with unknown effective dim. Choosing input dim.")
                effective_dim = f.dim

            info("*** Embedded TuRBO - Effective Dim ***")
            behavior = EmbeddedTuRBOBehavior(
                initial_base_length=l_init,
                min_base_length=l_min,
                max_base_length=l_max,
                acquisition_function=acquisition_function,
                embedding_type=bin_sizing_method,
                noise=noise_std,
            )
            gp_behaviour = GPBehaviour(
                mll_estimation=mle_optimization_method,
                n_initial_samples=multistart_samples,
                n_best_on_lhs_selection=multistart_after_samples,
                n_mle_training_steps=mle_training_steps,
            )
            conf_name = (
                f"embedded_turbo_effective_dim_{behavior}"
                f"_{gp_behaviour}"
            )
            run_dir = os.path.join(
                function_dir,
                str(adler32(conf_name.encode("utf-8"))),
            )
            embedded_turbo = EmbeddedTuRBO(
                f=f,
                target_dim=effective_dim,
                n_init=n_init,
                max_evals=max_evals,
                run_dir=run_dir,
                conf_name=conf_name,
                behavior=behavior,
                gp_behaviour=gp_behaviour,
            )
            run_and_plot(
                m=embedded_turbo,
                repetitions=repetitions,
                directory=run_dir,
            )
            del embedded_turbo

        if "embedded_turbo_2_effective_dim" == args.algorithm and (
                issubclass(type(f), EffectiveDimBenchmark)
                or issubclass(type(f), EffectiveDimBoTorchBenchmark)
        ):
            info(
                f"*** Embedded TuRBO- 2 * Effective Dim ***"
            )
            behavior = EmbeddedTuRBOBehavior(
                initial_base_length=l_init,
                min_base_length=l_min,
                max_base_length=l_max,
                acquisition_function=acquisition_function,
                embedding_type=bin_sizing_method,
                noise=noise_std,
            )
            gp_behaviour = GPBehaviour(
                mll_estimation=mle_optimization_method,
                n_initial_samples=multistart_samples,
                n_best_on_lhs_selection=multistart_after_samples,
                n_mle_training_steps=mle_training_steps,
            )
            conf_name = (
                f"embedded_turbo_2_times_effective_dim_{behavior}"
                f"_{gp_behaviour}"
            )
            run_dir = os.path.join(
                function_dir,
                str(adler32(conf_name.encode("utf-8"))),
            )
            embedded_turbo = EmbeddedTuRBO(
                f=f,
                target_dim=2 * f.effective_dim,
                n_init=n_init,
                max_evals=max_evals,
                run_dir=run_dir,
                conf_name=conf_name,
                behavior=behavior,
                gp_behaviour=gp_behaviour,
            )
            run_and_plot(
                m=embedded_turbo,
                repetitions=repetitions,
                directory=run_dir,
            )
            del embedded_turbo

        if "random_search" == args.algorithm:
            info(star_string("Random Search"))
            rs = RandomSearch(
                function=f,
                lower_bounds=llb,
                upper_bounds=uub,
                run_dir=os.path.join(function_dir, "random_search"),
                max_evals=max_evals,
                input_dim=input_dim
            )
            run_and_plot(
                m=rs,
                repetitions=repetitions,
                directory=os.path.join(function_dir, "random_search")
            )
