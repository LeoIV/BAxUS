import os
import tempfile
from unittest import TestCase, mock

import pandas as pd
import pytest
from parameterized import parameterized

from baxus.benchmark_runner import main
from baxus.util.exceptions import ArgumentError, EffectiveDimTooLargeException
from baxus.util.parsing import parse


def custom_name_func(testcase_func, param_num, param):
    return "_".join(param.args[1].split(" "))


def _load_data(*args, **kwargs):
    data_folder = "tests/data"
    if data_folder is None:
        data_folder = os.path.join(os.getcwd(), "data")
    data = pd.read_csv(
        os.path.join(data_folder, "slice_localization_data.csv.xz")
    ).to_numpy()
    X = data[:100, :385]
    y = data[:100, -1]
    return X, y


@mock.patch('baxus.embeddedturbo.lzma')
class BenchmarkRunnerTestSuite(TestCase):

    def _parse_func(self, args):
        """
        Use a temporary directory for runs instead of saving them to "results"
        :param args:
        :return:
        """
        directory_file_descriptor = tempfile.TemporaryDirectory()
        self.fd = directory_file_descriptor
        directory_name = directory_file_descriptor.name
        args = parse(args)
        args.results_dir = directory_name
        return args

    def tearDown(self) -> None:
        self.fd.cleanup()
        self.fd = None

    @parameterized.expand(
        [
            # 1: test ts
            (
                    "-id 6 -td 1 -n 2 -r 1 -m 10 --noise-std 0.1 -f branin2 -a baxus -l 0.02 --acquisition-function ts",
                    "test ei acquisition function",
            ),
            # 2: test ei
            (
                    "-id 6 -td 1 -n 2 -r 1 -m 10 --noise-std 0.1 -f branin2 -a baxus -l 0.02 --acquisition-function ei",
                    "test ei acquisition function",
            ),
            # 5: test multiple repetitions
            (
                    "-id 6 -td 2 -n 2 -r 2 -m 10 --noise-std 0.1 -f rosenbrock5 -a baxus",
                    "test multiple repetitions",
            ),
            # 6: test target dim 1
            (
                    "-id 6 -td 1 -n 2 -r 1 -m 10 --noise-std 0.1 -f rosenbrock5 -a baxus",
                    "test target dims",
            ),
            # 6: test target dim 2
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 10 --noise-std 0.1 -f rosenbrock5 -a baxus",
                    "test target dims",
            ),
            # 7: test input dim 6
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 5 --noise-std 0.1 -f rosenbrock5 -a baxus",
                    "test input dims",
            ),
            # 7: test input dim 7
            (
                    "-id 7 -td 2 -n 2 -r 1 -m 5 --noise-std 0.1 -f rosenbrock5 -a baxus",
                    "test input dims",
            ),
            # 13: test mle optimization multistart-gd
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 5 --noise-std 0.1 -f rosenbrock5 -a baxus "
                    "--mle-optimization multistart-gd",
                    "test mle optimization",
            ),
            # 13: test mle optimization sample-and-choose-best
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 5 --noise-std 0.1 -f rosenbrock5 -a baxus "
                    "--mle-optimization sample-and-choose-best",
                    "test mle optimization",
            ),
            # 14: test multistart after sample 2
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 5 --noise-std 0.1 -f rosenbrock5 -a baxus "
                    "--multistart-after-sample 2",
                    "test multistart after sample",
            ),
            # 14: test multistart after sample 3
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 5 --noise-std 0.1 -f rosenbrock5 -a baxus "
                    "--multistart-after-sample 3",
                    "test multistart after sample",
            ),
            # 15: check if baxus is running
            (
                    "-id 7 -td 2 -n 2 -r 1 -m 8 -f "
                    "hartmann6 -a baxus",
                    "test if baxus is running",
            ),
            # 15: check if embedded_turbo_target_dim is running
            (
                    "-id 7 -td 2 -n 2 -r 1 -m 8 -f "
                    "hartmann6 -a embedded_turbo_target_dim",
                    "test if embedded_turbo_target_dim is running",
            ),
            # 15: check if embedded_turbo_effective_dim is running
            (
                    "-id 7 -td 2 -n 2 -r 1 -m 8 -f "
                    "hartmann6 -a embedded_turbo_effective_dim",
                    "test if embedded_turbo_effective_dim is running",
            ),
            # 15: check if embedded_turbo_2_effective_dim is running
            (
                    "-id 7 -td 2 -n 2 -r 1 -m 8 -f "
                    "hartmann6 -a embedded_turbo_2_effective_dim",
                    "test if embedded_turbo_2_effective_dim is running",
            ),
            # 15: check if random_search is running
            (
                    "-id 7 -td 2 -n 2 -r 1 -m 8 -f "
                    "hartmann6 -a random_search",
                    "test if random_search is running",
            ),
            # 20: test multistart samples 11
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 10 -f rosenbrock5 -a baxus --multistart-samples 11",
                    "test multistart samples",
            ),
            # 20: test multistart samples 12
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 10 -f rosenbrock5 -a baxus --multistart-samples 12",
                    "test multistart samples",
            ),
            # 21: test mle training steps 31
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 5 -f rosenbrock5 -a baxus --mle-training-steps 31",
                    "test mle training steps",
            ),
            # 21: test mle training steps 32
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 5 -f rosenbrock5 -a baxus --mle-training-steps 32",
                    "test mle training steps",
            ),
        ],
        name_func=custom_name_func,
    )
    def test_sequence(self, _, conf: str, __):
        load_data_mock = mock.MagicMock(side_effect=_load_data)
        with mock.patch(
                "baxus.benchmarks.real_world_benchmarks.SVMBenchmark._load_data",
                load_data_mock,
        ):
            with mock.patch("baxus.benchmark_runner.parse",
                            mock.MagicMock(side_effect=self._parse_func)):
                main(conf.split(" "))

    @parameterized.expand(
        [
            (
                    "-id 6 -td 2 -n 2 -r 1 -m 4 --noise-std 0.1 -lmin 0.8 -lmax 0.4 -f rosenbrock5 "
                    "-a baxus",
                    "test min length larger than max length",
                    ArgumentError
            ),
            (
                    "-id 6 -td 10 -n 2 -r 1 -m 4 --noise-std 0.1 -f rosenbrock5 "
                    "-a baxus",
                    "test input dim smaller than target dim",
                    ArgumentError
            ),
            (
                    "-id 4 -td 2 -n 2 -r 1 -m 4 --noise-std 0.1 -f rosenbrock5 "
                    "-a baxus",
                    "test input dim too small for rosenbrock5",
                    EffectiveDimTooLargeException
            ),
            (
                    "-id 4 -td 2 -n 2 -r 1 -m 4 --noise-std 0.1 -f hartmann6 "
                    "-a baxus",
                    "test input dim too small for hartmann6",
                    EffectiveDimTooLargeException
            ),
            (
                    "-id 4 -td 2 -n 2 -r 1 -m 4 --noise-std 0.1 -f rosenbrock10 "
                    "-a baxus",
                    "test input dim too small for rosenbrock10",
                    EffectiveDimTooLargeException
            ),
            (
                    "-id 6 -td 2 -n 2 -r 1 --noise-std -0.3 -m 4 -f hartmann6 "
                    "-a baxus",
                    "test negative noise std",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 -bins 1 -a baxus",
                    "test one bin",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 -bins 0 -a baxus",
                    "test zero bins",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 -bins -1 -a baxus",
                    "test negative bins",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 --multistart-samples 0 -a baxus",
                    "test zero multistart samples",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 --multistart-samples -1 -a baxus",
                    "test negative multistart samples",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 --multistart-samples 5 --multistart-after-sample 6 -a baxus",
                    "test multistart after samples greater than initial multistart",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 --multistart-after-sample 0 -a baxus",
                    "test zero multistart after samples",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 --multistart-after-sample -1 -a baxus",
                    "test negative multistart after samples",
                    ArgumentError
            ),
            (
                    "-id 6 -td 2 -n 2 -m 4 -f hartmann6 --mle-training-steps -1 -a baxus",
                    "test negative mle training steps",
                    ArgumentError
            ),
        ],
        name_func=custom_name_func,
    )
    def test_illegal_configurations(self, __, conf: str, _, exception):
        load_data_mock = mock.MagicMock(side_effect=_load_data)
        with mock.patch(
                "baxus.benchmarks.real_world_benchmarks.SVMBenchmark._load_data",
                load_data_mock,
        ):
            with mock.patch("baxus.benchmark_runner.parse",
                            mock.MagicMock(side_effect=self._parse_func)):
                with pytest.raises(exception):
                    main(conf.split(" "))
