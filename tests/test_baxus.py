from unittest import TestCase

import numpy as np

from baxus.util.data_utils import join_data


class BAxUSTestSuite(TestCase):

    def test_join_data(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x1 = join_data(X=x, dims_and_bins={2: 2})
        self.assertTrue(np.allclose(x1[:, 2], x1[:, 3]))
        self.assertTrue(np.allclose(x[:, 2], x1[:, 3]))
        self.assertTrue(np.allclose(x[:, 0], x1[:, 0]))
        self.assertTrue(np.allclose(x[:, 1], x1[:, 1]))
        self.assertTrue(np.allclose(x[:, 2], x1[:, 2]))
        self.assertTrue(x1.shape == (2, 4))
        self.assertTrue(x.shape == (2, 3))
