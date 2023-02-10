import unittest

import ConfigSpace as CS

from lkauto.algorithms.funksvd import FunkSVD


class TestFunkSVD(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        features = 10
        funk_svd = FunkSVD(features)
        self.assertIsInstance(funk_svd, FunkSVD)
        self.assertEqual(features, funk_svd.features)

    def test_defaultConfigspace_GivenFunctionCalled_correctConfigSpaceReturnedExpected(self):
        cs = FunkSVD.get_default_configspace()
        params = cs.get_hyperparameters()
        self.assertIsInstance(cs, CS.ConfigurationSpace)
        self.assertTrue(any(
            param.name == "features" and param.default_value == 1000 and param.lower == 2 and param.upper == 10000 for
            param in params))
        self.assertTrue(any(
            param.name == "lrate" and param.default_value == 0.001 and param.lower == 0.0001 and param.upper == 0.01 for
            param in params))
        self.assertTrue(any(
            param.name == "reg" and param.default_value == 0.015 and param.lower == 0.001 and param.upper == 0.1 for
            param in params))
        self.assertTrue(any(
            param.name == "damping" and param.default_value == 5 and param.lower == 0.01 and param.upper == 1000 for
            param in params))
