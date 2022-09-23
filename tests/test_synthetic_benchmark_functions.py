import math
from unittest import TestCase

import numpy as np
import pytest

from baxus.benchmarks import AckleyEffectiveDim, RosenbrockEffectiveDim, LevyEffectiveDim, DixonPriceEffectiveDim, \
    BraninEffectiveDim, RastriginEffectiveDim, MichalewiczEffectiveDim, GriewankEffectiveDim, HartmannEffectiveDim, \
    RotatedHartmann6
from baxus.util.exceptions import EffectiveDimTooLargeException


class SyntheticBenchmarkFunctionsTestSuite(TestCase):
    def test_functions(self):
        funs = [AckleyEffectiveDim, RosenbrockEffectiveDim, LevyEffectiveDim, DixonPriceEffectiveDim,
                GriewankEffectiveDim, MichalewiczEffectiveDim, RastriginEffectiveDim, BraninEffectiveDim,
                HartmannEffectiveDim, ]
        for fun in funs:
            fun_instance = fun()
            assert fun_instance.dim == 200

            fun_instance = fun(dim=164)
            assert fun_instance.dim == 164
            fun_instance = fun(noise_std=3.0)
            f1 = fun_instance(np.zeros(200))
            f2 = fun_instance(np.zeros(200))
            self.assertNotAlmostEqual(f1, f2, places=5)
        funs = [BraninEffectiveDim, HartmannEffectiveDim, RosenbrockEffectiveDim]
        for fun in funs:
            fun_instance = fun()
            assert fun_instance.dim == 200

            fun_instance = fun(164)
            assert fun_instance.dim == 164

            fun_instance = fun(noise_std=3.0)
            f1 = fun_instance(np.zeros(200))
            f2 = fun_instance(np.zeros(200))
            self.assertNotAlmostEqual(f1, f2, places=5)

    def test_return_multiple_values(self):
        funs = [AckleyEffectiveDim, RosenbrockEffectiveDim, LevyEffectiveDim, DixonPriceEffectiveDim,
                GriewankEffectiveDim, MichalewiczEffectiveDim, RastriginEffectiveDim, BraninEffectiveDim,
                HartmannEffectiveDim, ]
        for fun in funs:
            fun_instance = fun()
            y = fun_instance(np.zeros((3, 200)))
            assert len(y) == 3
            assert np.allclose(y[0], y[1])
            assert np.allclose(y[1], y[2])

    def test_ackley_optimum(self):
        fun_instance = AckleyEffectiveDim()
        self.assertAlmostEqual(fun_instance(np.zeros(200)), 0.0, places=5)

    def test_ackley_bounds(self):
        fun_instance = AckleyEffectiveDim(47)
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=47, fill_value=-32.768))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=47, fill_value=32.768))
        )
        fun_instance = AckleyEffectiveDim()
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=200, fill_value=-32.768))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=200, fill_value=32.768))
        )

    def test_rosenbrock_optimum(self):
        fun_instance = RosenbrockEffectiveDim()
        self.assertAlmostEqual(fun_instance(np.ones(200)), 0.0, places=5)

    def test_rosenbrock_bounds(self):
        fun_instance = RosenbrockEffectiveDim(23)
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=23, fill_value=-5))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=23, fill_value=10))
        )
        fun_instance = RosenbrockEffectiveDim()
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=200, fill_value=-5))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=200, fill_value=10))
        )

    def test_levy_optimum(self):
        fun_instance = LevyEffectiveDim()
        self.assertAlmostEqual(fun_instance(np.ones(200)), 0.0, places=5)

    def test_levy_bounds(self):
        fun_instance = LevyEffectiveDim(34)
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=34, fill_value=-10))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=34, fill_value=10))
        )
        fun_instance = LevyEffectiveDim()
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=200, fill_value=-10))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=200, fill_value=10))
        )

    def test_dixon_price_optimum(self):
        fun_instance = DixonPriceEffectiveDim()
        point = np.array([2 ** (-(2 ** i - 2) / 2 ** i) for i in range(1, 201)])
        self.assertAlmostEqual(fun_instance(point), 0.0, places=5)

    def test_dixon_price_bounds(self):
        fun_instance = DixonPriceEffectiveDim(654)
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=654, fill_value=-10))
        )

        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=654, fill_value=10))
        )
        fun_instance = DixonPriceEffectiveDim()
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=200, fill_value=-10))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=200, fill_value=10))
        )

    def test_griewank_optimum(self):
        fun_instance = GriewankEffectiveDim()
        self.assertAlmostEqual(fun_instance(np.zeros(200)), 0.0, places=5)

    def test_griewank_price_bounds(self):
        fun_instance = GriewankEffectiveDim(123)
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=123, fill_value=-600))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=123, fill_value=600))
        )
        fun_instance = GriewankEffectiveDim()
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=200, fill_value=-600))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=200, fill_value=600))
        )

    def test_michalewicz_bounds(self):
        fun_instance = MichalewiczEffectiveDim(76)
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=76, fill_value=0))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=76, fill_value=math.pi))
        )
        fun_instance = MichalewiczEffectiveDim()
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=200, fill_value=0))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=200, fill_value=math.pi))
        )

    def test_rastrigin_optimum(self):
        fun_instance = RastriginEffectiveDim()
        self.assertAlmostEqual(fun_instance(np.zeros(200)), 0.0, places=5)

    def test_rastrigin_bounds(self):
        fun_instance = RastriginEffectiveDim(54)
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=54, fill_value=-5.12))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=54, fill_value=5.12))
        )
        fun_instance = RastriginEffectiveDim()
        self.assertTrue(
            np.allclose(fun_instance.lb_vec, np.full(shape=200, fill_value=-5.12))
        )
        self.assertTrue(
            np.allclose(fun_instance.ub_vec, np.full(shape=200, fill_value=5.12))
        )

    def test_effective_dim_benchmarks(self):
        hartmann6 = HartmannEffectiveDim()
        assert hartmann6.effective_dim == 6
        assert hartmann6.dim == 200
        branin2 = BraninEffectiveDim()
        assert branin2.effective_dim == 2
        assert branin2.dim == 200
        rb = RosenbrockEffectiveDim()
        assert rb.effective_dim == 10
        assert rb.dim == 200
        rb = RosenbrockEffectiveDim(effective_dim=54)
        assert rb.effective_dim == 54
        assert rb.dim == 200

        funs = [HartmannEffectiveDim, BraninEffectiveDim, RosenbrockEffectiveDim]
        for fun in funs:
            f_inst = fun(dim=145)
            assert f_inst.dim == 145

    def test_rosenbrock_eff_lower_than_true_dim(self):
        with pytest.raises(EffectiveDimTooLargeException):
            RosenbrockEffectiveDim(dim=23, effective_dim=24)

    def test_branin2_eff_lower_than_true_dim(self):
        with pytest.raises(EffectiveDimTooLargeException):
            BraninEffectiveDim(dim=1)

    def test_hartmann6_eff_lower_than_true_dim(self):
        with pytest.raises(EffectiveDimTooLargeException):
            HartmannEffectiveDim(dim=2)


class RotatedHartmann6Test(TestCase):
    def test_init(self):
        benchmark = RotatedHartmann6()
        self.assertEqual(benchmark.dim, 1000)

    def test_call_with_one_value_numpy_array(self):
        benchmark = RotatedHartmann6()
        input = np.random.uniform(low=benchmark.lb_vec, high=benchmark.ub_vec)
        results = benchmark(input)
        self.assertEqual(results.size, 1)

    def test_call_with_multiple_values_numpy_array(self):
        benchmark = RotatedHartmann6()
        input = np.random.uniform(
            low=benchmark.lb_vec, high=benchmark.ub_vec, size=(5, benchmark.dim)
        )
        results = benchmark(input)
        self.assertEqual(len(results), 5)

    def test_call_with_one_value_list(self):
        benchmark = RotatedHartmann6()
        input = np.random.uniform(low=benchmark.lb_vec, high=benchmark.ub_vec).tolist()
        results = benchmark(input)
        self.assertEqual(results.size, 1)

    def test_call_with_multiple_values_numpy_array(self):
        benchmark = RotatedHartmann6()
        input = np.random.uniform(
            low=benchmark.lb_vec, high=benchmark.ub_vec, size=(5, benchmark.dim)
        ).tolist()
        results = benchmark(input)
        self.assertEqual(len(results), 5)

    # TODO fix these tests
    # @expectedFailure
    # def test_out_of_lower_bounds(self):
    #    benchmark = RotatedHartmann6()
    #    input = np.random.uniform(
    #        low=benchmark.lb_vec - 5.0,
    #        high=benchmark.lb_vec - 1.0,
    #        size=(5, benchmark.dim),
    #    )
    #    benchmark(input)

    # @expectedFailure
    # def test_out_of_upper_bounds(self):
    #    benchmark = RotatedHartmann6()
    #    input = np.random.uniform(
    #        low=benchmark.ub_vec + 1.0,
    #        high=benchmark.lb_vec + 5.0,
    #        size=(5, benchmark.dim),
    #    )
    #    benchmark(input)
