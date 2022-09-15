from unittest import TestCase, mock

import numpy as np
import pytest
import torch
from gpytorch.kernels import ScaleKernel, MaternKernel

from baxus.gp import train_gp
from baxus.kernels import (
    KernelType,
    KPLSRBFKernel,
    PLSContainer,
    KPLSMaternKernel,
    KPLSKRBFKernel,
)
from baxus.util.behaviors.gp_configuration import GPBehaviour, MLLEstimation


class GPTestSuite(TestCase):
    def setUp(self) -> None:
        X = np.random.randn(100, 23)  # 100 samples, 23 feature_maps_vae
        y = np.random.randn(100)

        X_local = X[70:]
        y_local = y[70:]

        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

        self.X_local = torch.tensor(X_local)
        self.y_local = torch.tensor(y_local)

    def test_gp_matern_no_pls(self):
        # train_gp with use_pls set to False should return MaternKernel
        gaussian_process, pls_container, _ = train_gp(
            train_x=self.X_local,
            train_y=self.y_local,
            global_x=self.X,
            global_y=self.y,
            use_ard=True,
            gp_behaviour=GPBehaviour(),
            use_pls=False,
            pls=None,
            kernel_type=KernelType.KPLSK_RBF,
        )

        self.assertIsInstance(gaussian_process.covar_module, ScaleKernel)
        self.assertIsInstance(gaussian_process.covar_module.base_kernel, MaternKernel)
        self.assertIsNone(pls_container)

    def test_gp_kpls_pls(self):
        gaussian_process, pls_container, _ = train_gp(
            train_x=self.X_local,
            train_y=self.y_local,
            global_x=self.X,
            global_y=self.y,
            use_ard=True,
            gp_behaviour=GPBehaviour(),
            use_pls=True,
            pls=None,
            kernel_type=KernelType.KPLS_RBF,
            n_pls_components=2,
        )

        self.assertIsInstance(gaussian_process.covar_module, ScaleKernel)
        self.assertIsInstance(gaussian_process.covar_module.base_kernel, KPLSRBFKernel)
        self.assertIsInstance(pls_container, PLSContainer)
        self.assertEqual(len(gaussian_process.lengthscales), 23)
        self.assertEqual(len(gaussian_process.covar_module.base_kernel.eta), 23)
        self.assertEqual(
            len(
                gaussian_process.covar_module.base_kernel.lengthscale.detach()
                    .numpy()
                    .squeeze()
            ),
            2,
        )

    def test_gp_kplsk_pls(self):
        gaussian_process, pls_container, _ = train_gp(
            train_x=self.X_local,
            train_y=self.y_local,
            global_x=self.X,
            global_y=self.y,
            use_ard=True,
            gp_behaviour=GPBehaviour(),
            use_pls=True,
            pls=None,
            kernel_type=KernelType.KPLSK_RBF,
            n_pls_components=2,
        )

        self.assertIsInstance(gaussian_process.covar_module, ScaleKernel)
        self.assertIsInstance(gaussian_process.covar_module.base_kernel, KPLSKRBFKernel)
        self.assertIsInstance(pls_container, PLSContainer)
        self.assertEqual(len(gaussian_process.lengthscales), 23)
        self.assertEqual(len(gaussian_process.covar_module.base_kernel.eta), 23)
        self.assertEqual(
            len(
                gaussian_process.covar_module.base_kernel.eta.detach().numpy().squeeze()
            ),
            23,
        )

    def test_gp_kpls_matern_pls(self):
        gaussian_process, pls_container, _ = train_gp(
            train_x=self.X_local,
            train_y=self.y_local,
            global_x=self.X,
            global_y=self.y,
            use_ard=True,
            gp_behaviour=GPBehaviour(),
            use_pls=True,
            pls=None,
            kernel_type=KernelType.KPLS_MATERN,
            n_pls_components=2,
        )

        self.assertIsInstance(gaussian_process.covar_module, ScaleKernel)
        self.assertIsInstance(
            gaussian_process.covar_module.base_kernel, KPLSMaternKernel
        )
        self.assertIsInstance(pls_container, PLSContainer)
        self.assertEqual(len(gaussian_process.lengthscales), 23)
        self.assertEqual(len(gaussian_process.covar_module.base_kernel.eta), 23)
        self.assertEqual(
            len(
                gaussian_process.covar_module.base_kernel.lengthscale.detach()
                    .numpy()
                    .squeeze()
            ),
            2,
        )

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_kpls_kplsk_transition(
            self, _, m2: mock.MagicMock
    ):  # inverse order, last is GP Mock
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            gaussian_process, pls_container, _ = train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(),
                use_pls=True,
                pls=None,
                kernel_type=KernelType.KPLSK_RBF,
                n_pls_components=2,
            )
            m2.assert_any_call(
                train_x=mock.ANY,
                train_y=mock.ANY,
                global_x=mock.ANY,
                global_y=mock.ANY,
                lengthscale_constraint=mock.ANY,
                outputscale_constraint=mock.ANY,
                likelihood=mock.ANY,
                ard_dims=mock.ANY,
                use_latent_space_kernel=True,
                n_pls_components=2,
                pls=None,
                kernel_type=KernelType.KPLS_RBF,
                mu=mock.ANY,
                sigma=mock.ANY,
            )
            m2.assert_any_call(
                train_x=mock.ANY,
                train_y=mock.ANY,
                global_x=mock.ANY,
                global_y=mock.ANY,
                lengthscale_constraint=mock.ANY,
                outputscale_constraint=mock.ANY,
                likelihood=mock.ANY,
                ard_dims=mock.ANY,
                use_latent_space_kernel=True,
                n_pls_components=2,
                pls=None,
                kernel_type=KernelType.KPLSK_RBF,
                mu=mock.ANY,
                sigma=mock.ANY,
            )
            assert m2.call_count == 2

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_no_pls_kernel_scale_constraints(self, _, gp_mock: mock.MagicMock):
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            gaussian_process, pls_container, _ = train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(),
                use_pls=False,
                pls=None,
                kernel_type=KernelType.KPLSK_RBF,
            )
            lengthscale_constraint = gp_mock.call_args_list[0][1][
                "lengthscale_constraint"
            ]
            outputscale_constraint = gp_mock.call_args_list[0][1][
                "outputscale_constraint"
            ]

            self.assertTrue(outputscale_constraint.lower_bound == 0.5)
            self.assertTrue(outputscale_constraint.upper_bound == 2.0)

            self.assertTrue(lengthscale_constraint.lower_bound == 0.005)
            self.assertTrue(lengthscale_constraint.upper_bound == 2.0)

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_kpls_matern_kernel_scale_constraints(self, _, gp_mock: mock.MagicMock):
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            gaussian_process, pls_container, _ = train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(),
                use_pls=True,
                pls=None,
                kernel_type=KernelType.KPLS_MATERN,
            )
            lengthscale_constraint = gp_mock.call_args_list[0][1][
                "lengthscale_constraint"
            ]
            outputscale_constraint = gp_mock.call_args_list[0][1][
                "outputscale_constraint"
            ]

            self.assertTrue(outputscale_constraint.lower_bound == 0.5)
            self.assertTrue(outputscale_constraint.upper_bound == 2.0)

            self.assertTrue(lengthscale_constraint.lower_bound == 0.005)
            self.assertTrue(lengthscale_constraint.upper_bound == 707)

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_kpls_rbf_kernel_scale_constraints(self, _, gp_mock: mock.MagicMock):
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            gaussian_process, pls_container, _ = train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(),
                use_pls=True,
                pls=None,
                kernel_type=KernelType.KPLS_RBF,
            )
            lengthscale_constraint = gp_mock.call_args_list[0][1][
                "lengthscale_constraint"
            ]
            outputscale_constraint = gp_mock.call_args_list[0][1][
                "outputscale_constraint"
            ]

            self.assertTrue(outputscale_constraint.lower_bound == 0.5)
            self.assertTrue(outputscale_constraint.upper_bound == 2.0)

            self.assertTrue(lengthscale_constraint.lower_bound == 0.005)
            self.assertTrue(lengthscale_constraint.upper_bound == 707)

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_kpls_rbf_kernel_scale_constraints(self, _, gp_mock: mock.MagicMock):
        def _pick_best_from_configurations(
                initializers, model, train_x, train_y, mll, n_best=1
        ):
            return np.array(initializers)[:n_best].tolist()

        def _mle_optimization(
                initializer, model, num_steps, kernel_type, train_x, train_y, mll
        ):
            return model.get_state_dict(), 0.3

        with mock.patch(
                "turbo.gp.pick_best_from_configurations",
                mock.MagicMock(side_effect=_pick_best_from_configurations),
        ):
            with mock.patch(
                    "turbo.gp.mle_optimization",
                    mock.MagicMock(side_effect=_mle_optimization),
            ):
                gaussian_process, pls_container, _ = train_gp(
                    train_x=self.X_local,
                    train_y=self.y_local,
                    global_x=self.X,
                    global_y=self.y,
                    use_ard=True,
                    gp_behaviour=GPBehaviour(),
                    use_pls=True,
                    pls=None,
                    kernel_type=KernelType.KPLSK_RBF,
                )
                lengthscale_constraint_1 = gp_mock.call_args_list[0][1][
                    "lengthscale_constraint"
                ]
                lengthscale_constraint_2 = gp_mock.call_args_list[1][1][
                    "lengthscale_constraint"
                ]
                outputscale_constraint = gp_mock.call_args_list[0][1][
                    "outputscale_constraint"
                ]

                self.assertTrue(outputscale_constraint.lower_bound == 0.5)
                self.assertTrue(outputscale_constraint.upper_bound == 2.0)

                self.assertTrue(lengthscale_constraint_1.lower_bound == 0.005)
                self.assertTrue(lengthscale_constraint_1.upper_bound == 707)

                self.assertTrue(lengthscale_constraint_2.lower_bound == 0.005)
                self.assertTrue(lengthscale_constraint_2.upper_bound == 2.0)

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_no_pls_kernel_lengthscale(self, _, gp_mock):
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(
                    mll_estimation=MLLEstimation.MULTI_START_GRADIENT_DESCENT,
                    n_initial_samples=1,
                ),
                use_pls=False,
                pls=None,
                kernel_type=KernelType.KPLSK_RBF,
            )
            mock_calls = mle_mock.mock_calls
            mle_calls = [m for m in mock_calls if "initializer" in m[2]]
            self.assertEqual(1, len(mle_calls))
            mle_call = mle_calls[0]
            initializer = mle_call[2]["initializer"]
            gp_to_to_initialize = mock.MagicMock()
            initializer(gp_to_to_initialize)
            initializer_calls = [
                m for m in gp_to_to_initialize.mock_calls if "initialize" in m[0]
            ]
            self.assertEqual(len(initializer_calls), 1)
            initialize_call = initializer_calls[0]
            self.assertEqual(
                initialize_call[2]["covar_module.base_kernel.lengthscale"], 0.5
            )

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_kpls_matern_kernel_lengthscale(self, mk, gp):
        values = ({}, 0.3)

        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            gaussian_process, pls_container, _ = train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(),
                use_pls=True,
                pls=None,
                n_pls_components=2,
                kernel_type=KernelType.KPLS_MATERN,
            )
            kernel = gaussian_process.covar_module.base_kernel
            self.assertTrue(
                np.allclose(kernel.lengthscale.detach().numpy().squeeze(), 22.0)
            )

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_kpls_rbf_kernel_lengthscale(self, _, gp_mock: mock.MagicMock):
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(
                    mll_estimation=MLLEstimation.MULTI_START_GRADIENT_DESCENT,
                    n_initial_samples=1,
                ),
                use_pls=True,
                pls=None,
                kernel_type=KernelType.KPLS_RBF,
            )
            mock_calls = mle_mock.mock_calls
            mle_calls = [m for m in mock_calls if "initializer" in m[2]]
            self.assertEqual(1, len(mle_calls))
            mle_call = mle_calls[0]
            initializer = mle_call[2]["initializer"]
            gp_to_to_initialize = mock.MagicMock()
            initializer(gp_to_to_initialize)
            initializer_calls = [
                m for m in gp_to_to_initialize.mock_calls if "initialize" in m[0]
            ]
            assert len(initializer_calls) == 1
            initializer_call = initializer_calls[0]
            self.assertEqual(
                initializer_call[2]["covar_module.base_kernel.lengthscale"], 22
            )

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_kplsk_rbf_kernel_lengthscale(self, _, gp_mock: mock.MagicMock):
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(
                    mll_estimation=MLLEstimation.MULTI_START_GRADIENT_DESCENT,
                    n_initial_samples=1,
                ),
                use_pls=True,
                pls=None,
                kernel_type=KernelType.KPLSK_RBF,
            )
            mock_calls = mle_mock.mock_calls
            mle_calls = [m for m in mock_calls if "initializer" in m[2]]
            self.assertEqual(2, len(mle_calls))
            first_mle_call = mle_calls[0]
            first_initializer = first_mle_call[2]["initializer"]
            gp_to_to_initialize = mock.MagicMock()
            first_initializer(gp_to_to_initialize)
            initializer_calls = [
                m for m in gp_to_to_initialize.mock_calls if "initialize" in m[0]
            ]
            self.assertEqual(len(initializer_calls), 1)
            initialize_call_0 = initializer_calls[0]
            self.assertEqual(
                initialize_call_0[2]["covar_module.base_kernel.lengthscale"], 22
            )
            second_mle_call = mle_calls[1]
            second_initializer = second_mle_call[2]["initializer"]
            gp_to_to_initialize.reset_mock()
            second_initializer(gp_to_to_initialize)
            initializer_calls = [
                m for m in gp_to_to_initialize.mock_calls if "initialize" in m[0]
            ]
            self.assertEqual(len(initializer_calls), 1)
            initialize_call_0 = initializer_calls[0]
            self.assertEqual(1, len(initialize_call_0[2].keys()))
            self.assertTrue("likelihood.noise" in list(initialize_call_0[2].keys()))

    def test_kpls_kernel_negative_lengthscale_dim(self):
        with pytest.raises(RuntimeError):
            gaussian_process, pls_container, _ = train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(),
                use_pls=True,
                pls=None,
                kernel_type=KernelType.KPLSK_RBF,
            )

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_number_of_steps(self, _, gp_mock: mock.MagicMock):
        values = ({}, 0.3)
        mle_mock = mock.MagicMock(return_value=values)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            for ms_samples in range(1, 5):
                train_gp(
                    train_x=self.X_local,
                    train_y=self.y_local,
                    global_x=self.X,
                    global_y=self.y,
                    use_ard=True,
                    gp_behaviour=GPBehaviour(
                        mll_estimation=MLLEstimation.MULTI_START_GRADIENT_DESCENT,
                        n_initial_samples=ms_samples,
                    ),
                    use_pls=True,
                    pls=None,
                    kernel_type=KernelType.KPLS_MATERN,
                )
                self.assertEqual(ms_samples, mle_mock.call_count)
                mle_mock.reset_mock()

    @mock.patch("turbo.gp.GP")
    @mock.patch("turbo.gp.ExactMarginalLogLikelihood")
    def test_load_best_state_dict(self, mll_mock, gp_mock: mock.MagicMock):
        values = {
            0: ({0.3: 0.3}, 0.3),
            1: ({0.4: 0.4}, 0.4),
            2: ({0.5: 0.5}, 0.5),
            3: ({0.2: 0.2}, 0.2),
            4: ({0.35: 0.35}, 0.35),
        }
        i = 0

        def side_effect(*args, **kwargs):
            nonlocal i
            val = values[i]
            i = i + 1
            return val

        mle_mock = mock.MagicMock(side_effect=side_effect)
        with mock.patch("turbo.gp.mle_optimization", mle_mock):
            train_gp(
                train_x=self.X_local,
                train_y=self.y_local,
                global_x=self.X,
                global_y=self.y,
                use_ard=True,
                gp_behaviour=GPBehaviour(
                    mll_estimation=MLLEstimation.MULTI_START_GRADIENT_DESCENT,
                    n_initial_samples=5,
                ),
                use_pls=True,
                pls=None,
                kernel_type=KernelType.KPLS_MATERN,
            )
            self.assertEqual(5, mle_mock.call_count)
            state_dict_calls = [
                c for c in gp_mock.mock_calls if "load_state_dict" in c[0]
            ]
            self.assertEqual(1, len(state_dict_calls))
            state_dict_call_param = state_dict_calls[0][1]
            self.assertDictEqual({0.2: 0.2}, state_dict_call_param[0])
