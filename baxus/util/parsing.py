import functools
from argparse import ArgumentParser, Namespace

from baxus.benchmarks.synthetic_benchmark_functions import (
    RosenbrockEffectiveDim,
    BraninEffectiveDim,
    HartmannEffectiveDim,
    RotatedHartmann6,
    AckleyEffectiveDim,
    LevyEffectiveDim,
    GriewankEffectiveDim,
    DixonPriceEffectiveDim,
    MichalewiczEffectiveDim,
    RastriginEffectiveDim,
    ShiftedAckley10,
)
from baxus.util.acquisition_function_types import AcquisitionFunctionType
from baxus.util.behaviors.gp_configuration import MLLEstimation
from baxus.util.behaviors.embedding_configuration import EmbeddingType


def parse(args):
    """
    Define a CLI parser and parse command line arguments

    Args:
        args: command line arguments

    Returns:
        Namespace: parsed command line arguments

    """
    parser = ArgumentParser()
    required_named = parser.add_argument_group("required named arguments")
    parser.add_argument(
        "-id",
        "--input-dim",
        type=int,
        default=100,
        help="Input dimensionality",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="baxus",
        choices=["baxus", "embedded_turbo_target_dim", "embedded_turbo_effective_dim",
                 "embedded_turbo_2_effective_dim", "random_search"],
        help="The algorithm"
    )

    parser.add_argument(
        "-l",
        "--initial-baselength",
        type=float,
        default=0.8,
        help="The initial base length.",
    )
    parser.add_argument(
        "-lmin",
        "--min-baselength",
        type=float,
        default=0.5 ** 7,
        help="The minimum base length.",
    )
    parser.add_argument(
        "-lmax",
        "--max-baselength",
        type=float,
        default=1.6,
        help="The maximum base length.",
    )

    parser.add_argument(
        "-td",
        "--target-dim",
        type=int,
        default=10,
        help="Target dimensionality",
    )

    parser.add_argument(
        "-n", "--n-init", type=int, help="Number of initial sampling points. Default: target dimensionality + 1."
    )

    parser.add_argument(
        "-r",
        "--num-repetitions",
        default=1,
        type=int,
        help="Number of independent repetitions of each run.",
    )

    parser.add_argument(
        "-m",
        "--max-evals",
        type=int,
        default=300,
        help="Max number of evaluations of each algorithm.",
    )
    parser.add_argument(
        "--noise-std",
        default=0.0,
        type=float,
        help="Standard deviation of the noise of the objective function.",
    )

    required_named.add_argument(
        "-f",
        "--function",
        choices=[
            "hartmann6",
            "branin2",
            "rosenbrock2",
            "rosenbrock5",
            "rosenbrock10",
            "rosenbrock15",
            "ackley2",
            "shiftedackley10",
            "ackley1",
            "rosenbrock-domain-fixed",
            "levy2",
            "levy1",
            "levy43",
            "dixonprice2",
            "griewank2",
            "griewank1",
            "michalewicz2",
            "michalewicz15",
            "michalewicz1",
            "rastrigin2",
            "rastrigin1",
            "svm",
            "lasso-leukemia",
            "lasso-breastcancer",
            "lasso-dna",
            "lasso-rcv1",
            "lasso-diabetes",
            "lasso-simple",
            "lasso-medium",
            "lasso-high",
            "lasso-hard",
            "mopta08",
            "hartmann6in1000_rotated",
            "rosenbrock5in1000_rotated",
        ],
        required=True,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory to store results in",
    )
    parser.add_argument(
        "--run-description",
        type=str,
        default="",
        help="Short description that will be added to the run directory",
    )
    parser.add_argument(
        "-bins", "--new-bins-on-split", type=int, default=3
    )

    parser.add_argument(
        "--multistart-samples",
        help="Number of multistart samples for the MLE GD optimization. Samples will be drawn from "
             "latin hypercube (if more than 1, otherwise the default value will be used",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--multistart-after-sample",
        type=int,
        default=10,
        help="Only recognized for '--mle-optimization sample-and-choose-best'. Number of multi-start "
             "gradient descent optimization out of the '--multistart-samples best ones.",
    )

    parser.add_argument(
        "--mle-optimization",
        choices=["multistart-gd", "sample-and-choose-best"],
        type=str,
        default="sample-and-choose-best",
        help="'multistart-gd': sample --multistart-samples different starting points for the hyperparameters and start "
             "gradient descent for each of them. 'sample-and-choose-best': evaluate -mss many "
             "initial configurations and start ",
    )
    parser.add_argument(
        "--mle-training-steps",
        type=int,
        default=50,
        help="Number of GD steps in MLE maximization.",
    )

    parser.add_argument(
        "--acquisition-function",
        type=str,
        default="ts",
        choices=["ts", "ei"],
        help="The acquisition functions to use. Either 'ei' or 'ts'"
    )

    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["hesbo", "baxus"],
        default="baxus",
        help="How to choose the bin sizes for the HeSBO embedding. 'hesbo': original HeSBO choice, pick "
             "one target dimension for each input dimension at random. 'baxus': ensure (almost) uniform"
             " bin sizes.",
    )

    parser.add_argument(
        "--budget-until-input-dim",
        type=int,
        default=0,
        help="The evaluation budget after which we reach the input dimension under the assumption that " \
             "we always fail in making progress."
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to print debug messages"
    )

    parser.add_argument(
        "--adjust-initial-target-dimension", action="store_true", help="Whether to adjust the initial target dimension"
                                                                       " such that the final split is as close "
                                                                       "as possible to the "
                                                                       "ambient dimension for BAxUS."
    )

    pars = parser.parse_args(args)

    # load required benchmarks
    benchmark_loader(pars.function, pars)
    return pars


acquisition_function_mapper = {
    "ts": AcquisitionFunctionType.THOMPSON_SAMPLING,
    "ei": AcquisitionFunctionType.EXPECTED_IMPROVEMENT,
}

mle_optimization_mapper = {
    "multistart-gd": MLLEstimation.MULTI_START_GRADIENT_DESCENT,
    "sample-and-choose-best": MLLEstimation.LHS_PICK_BEST_START_GD,
}

embedding_type_mapper = {
    "baxus": EmbeddingType.BAXUS,
    "hesbo": EmbeddingType.HESBO,
}
_fun_mapper = {}


def benchmark_loader(bench: str, args: Namespace):
    """
    Import the required implementation of a benchmark. We use this class to avoid imports of benchmarks that require
    optional dependencies.

    Args:
        bench: the benchmark name
        args: the parsed command line arguments

    Returns:
        None. Just import the benchmark implementation.

    """

    if bench == "lasso-leukemia":
        from baxus.benchmarks.real_world_benchmarks import LassoLeukemiaBenchmark

        _fun_mapper[bench] = LassoLeukemiaBenchmark

    if bench == "lasso-breastcancer":
        from baxus.benchmarks.real_world_benchmarks import LassoBreastCancerBenchmark

        _fun_mapper[bench] = LassoBreastCancerBenchmark

    if bench == "lasso-dna":
        from baxus.benchmarks.real_world_benchmarks import LassoDNABenchmark

        _fun_mapper[bench] = LassoDNABenchmark

    if bench == "lasso-diabetes":
        from baxus.benchmarks.real_world_benchmarks import LassoDiabetesBenchmark

        _fun_mapper[bench] = LassoDiabetesBenchmark

    if bench == "lasso-rcv1":
        from baxus.benchmarks.real_world_benchmarks import LassoRCV1Benchmark

        _fun_mapper[bench] = LassoRCV1Benchmark
    if bench == "lasso-simple":
        from baxus.benchmarks.real_world_benchmarks import LassoSimpleBenchmark

        _fun_mapper[bench] = LassoSimpleBenchmark
    if bench == "lasso-medium":
        from baxus.benchmarks.real_world_benchmarks import LassoMediumBenchmark

        _fun_mapper[bench] = LassoMediumBenchmark
    if bench == "lasso-high":
        from baxus.benchmarks.real_world_benchmarks import LassoHighBenchmark

        _fun_mapper[bench] = LassoHighBenchmark
    if bench == "lasso-hard":
        from baxus.benchmarks.real_world_benchmarks import LassoHardBenchmark

        _fun_mapper[bench] = LassoHardBenchmark

    if bench == "mopta08":
        from baxus.benchmarks.real_world_benchmarks import MoptaSoftConstraints

        _fun_mapper[bench] = MoptaSoftConstraints

    if bench == "svm":
        from baxus.benchmarks.real_world_benchmarks import SVMBenchmark

        _fun_mapper[bench] = SVMBenchmark


def fun_mapper():
    """
    Map benchmark names to their implementation.

    Returns:
        dict: a mapping of benchmark names to their (partially initialized) classes

    """
    return {
        **{
            "hartmann6": functools.partial(HartmannEffectiveDim, effective_dim=6),
            "branin2": functools.partial(BraninEffectiveDim, effective_dim=2),
            "rosenbrock2": functools.partial(
                RosenbrockEffectiveDim, effective_dim=2
            ),
            "rosenbrock5": functools.partial(
                RosenbrockEffectiveDim, effective_dim=5
            ),
            "rosenbrock10": functools.partial(
                RosenbrockEffectiveDim, effective_dim=10
            ),
            "rosenbrock15": functools.partial(
                RosenbrockEffectiveDim, effective_dim=15
            ),
            "ackley2": functools.partial(AckleyEffectiveDim, effective_dim=2),
            "shiftedackley10": ShiftedAckley10,
            "ackley1": functools.partial(AckleyEffectiveDim, effective_dim=1),
            "levy2": functools.partial(LevyEffectiveDim, effective_dim=2),
            "levy43": functools.partial(LevyEffectiveDim, effective_dim=43),
            "levy1": functools.partial(LevyEffectiveDim, effective_dim=1),
            "dixonprice2": functools.partial(DixonPriceEffectiveDim, effective_dim=2),
            "griewank2": functools.partial(GriewankEffectiveDim, effective_dim=2),
            "griewank1": functools.partial(GriewankEffectiveDim, effective_dim=1),
            "michalewicz2": functools.partial(MichalewiczEffectiveDim, effective_dim=2),
            "michalewicz1": functools.partial(MichalewiczEffectiveDim, effective_dim=1),
            "michalewicz15": functools.partial(
                MichalewiczEffectiveDim, effective_dim=15
            ),
            "rastrigin2": functools.partial(RastriginEffectiveDim, effective_dim=2),
            "rastrigin1": functools.partial(RastriginEffectiveDim, effective_dim=1),
            "hartmann6in1000_rotated": RotatedHartmann6,
        },
        **_fun_mapper,
    }
