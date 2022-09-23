import os
import shutil
from typing import List, Union
from unittest import TestCase, mock

import numpy as np
import pandas as pd
import pytest

from baxus.benchmarks import Benchmark, SyntheticBenchmark, SVMBenchmark, MoptaSoftConstraints
from baxus.util.exceptions import OutOfBoundsException, BoundsMismatchException


class BenchmarkTestSuite(TestCase):
    def test_init(self):
        benchmark = Benchmark(100, np.ones(100), np.zeros(100), noise_std=0)
        self.assertEqual(100, benchmark.dim)
        self.assertTrue(np.allclose(np.ones(100), benchmark.ub_vec))
        self.assertTrue(np.allclose(np.zeros(100), benchmark.lb_vec))
        self.assertEqual("Benchmark", benchmark.fun_name)

    def test_fun(self):
        with pytest.raises(NotImplementedError):
            benchmark = Benchmark(100, np.ones(100), np.zeros(100), noise_std=0)
            benchmark(3)

    def test_lower_bound_larger_than_upper_bound(self):
        with pytest.raises(OutOfBoundsException):
            benchmark = Benchmark(100, np.zeros(100), np.ones(100), noise_std=0)

    def test_one_bound_dim_unequal_dim(self):
        with pytest.raises(BoundsMismatchException):
            benchmark = Benchmark(100, np.ones(99), np.zeros(100), noise_std=0)

    def test_both_bounds_dim_unequal_dim(self):
        with pytest.raises(BoundsMismatchException):
            benchmark = Benchmark(100, np.ones(99), np.zeros(99), noise_std=0)


class SyntheticBenchmarkTestSuite(TestCase):
    def setUp(self) -> None:
        class SyntheticBenchmarkInstance(SyntheticBenchmark):
            """
            Simple class to allow for instantiation
            """

            def __init__(self, dim: int, ub: np.ndarray, lb: np.ndarray):
                super().__init__(dim, ub, lb, noise_std=0)

            def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
                return super().__call__(x)

        self.synthetic_benchmark = SyntheticBenchmarkInstance

        return super().setUp()

    def test_init(self):
        benchmark = self.synthetic_benchmark(100, np.ones(100), np.zeros(100))
        self.assertEqual(100, benchmark.dim)
        self.assertTrue(np.allclose(np.ones(100), benchmark.ub_vec))
        self.assertTrue(np.allclose(np.zeros(100), benchmark.lb_vec))
        self.assertEqual("SyntheticBenchmarkInstance", benchmark.fun_name)

    def test_lower_bound_larger_than_upper_bound(self):
        with pytest.raises(OutOfBoundsException):
            benchmark = self.synthetic_benchmark(100, np.zeros(100), np.ones(100))

    def test_one_bound_dim_unequal_dim(self):
        with pytest.raises(BoundsMismatchException):
            benchmark = self.synthetic_benchmark(100, np.ones(99), np.zeros(100))

    def test_both_bounds_dim_unequal_dim(self):
        with pytest.raises(BoundsMismatchException):
            benchmark = self.synthetic_benchmark(100, np.ones(99), np.zeros(99))

    def test_optimal_value(self):
        benchmark = self.synthetic_benchmark(100, np.ones(100), np.zeros(100))
        self.assertIsNone(benchmark.optimal_value)


class SVMBenchmarkTestSuite(TestCase):
    def setUp(self) -> None:
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

        load_data_mock = mock.MagicMock(side_effect=_load_data)
        with mock.patch(
                "baxus.benchmarks.real_world_benchmarks.SVMBenchmark._load_data",
                load_data_mock,
        ):
            self.benchmark = SVMBenchmark()

    def test_init(self):
        self.assertEqual(self.benchmark.dim, 388)

    def test_call_with_one_value_numpy_array(self):
        input = np.random.uniform(low=self.benchmark.lb_vec, high=self.benchmark.ub_vec)
        results = self.benchmark(input)
        self.assertEqual(results.size, 1)

    def test_call_with_multiple_values_numpy_array(self):
        input = np.random.uniform(
            low=self.benchmark.lb_vec,
            high=self.benchmark.ub_vec,
            size=(5, self.benchmark.dim),
        )
        results = self.benchmark(input)
        self.assertEqual(len(results), 5)

    def test_call_with_one_value_list(self):
        input = np.random.uniform(
            low=self.benchmark.lb_vec, high=self.benchmark.ub_vec
        ).tolist()
        results = self.benchmark(input)
        self.assertEqual(results.size, 1)

    def test_call_with_multiple_values_numpy_array(self):
        input = np.random.uniform(
            low=self.benchmark.lb_vec,
            high=self.benchmark.ub_vec,
            size=(5, self.benchmark.dim),
        ).tolist()
        results = self.benchmark(input)
        self.assertEqual(len(results), 5)


class MoptaBenchmarkTestSuite(TestCase):
    def test_init(self):
        benchmark = MoptaSoftConstraints()
        self.assertEqual(benchmark.dim, 124)

    def test_call_with_one_value_numpy_array(self):
        benchmark = MoptaSoftConstraints()
        input = np.random.uniform(low=benchmark.lb_vec, high=benchmark.ub_vec)
        results = benchmark(input)
        self.assertEqual(results.size, 1)

    def test_call_with_multiple_values_numpy_array(self):
        benchmark = MoptaSoftConstraints()
        input = np.random.uniform(
            low=benchmark.lb_vec, high=benchmark.ub_vec, size=(5, benchmark.dim)
        )
        results = benchmark(input)
        self.assertEqual(len(results), 5)

    def test_call_with_one_value_list(self):
        benchmark = MoptaSoftConstraints()
        input = np.random.uniform(low=benchmark.lb_vec, high=benchmark.ub_vec).tolist()
        results = benchmark(input)
        self.assertEqual(results.size, 1)

    def test_call_with_multiple_values_numpy_array(self):
        benchmark = MoptaSoftConstraints()
        input = np.random.uniform(
            low=benchmark.lb_vec, high=benchmark.ub_vec, size=(5, benchmark.dim)
        ).tolist()
        results = benchmark(input)
        self.assertEqual(len(results), 5)

    def test_write_to_generated_dir(self):
        benchmark = MoptaSoftConstraints()
        tmp_dir = benchmark.directory_name
        input = np.random.uniform(low=benchmark.lb_vec, high=benchmark.ub_vec)
        benchmark(input)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "input.txt")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "output.txt")))

    def test_write_to_given_dir(self):
        tmp_dir = "".join([str(x) for x in np.random.randint(0, 10, 10)])
        benchmark = MoptaSoftConstraints(tmp_dir)
        input = np.random.uniform(low=benchmark.lb_vec, high=benchmark.ub_vec)
        benchmark(input)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "input.txt")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "output.txt")))
        shutil.rmtree(tmp_dir)
