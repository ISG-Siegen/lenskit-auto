import unittest

import ConfigSpace as CS

from lkauto.algorithms.als import ImplicitMF, BiasedMF


class TestImplicitMF(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        expected_features = 10
        implicit_mf = ImplicitMF(expected_features)
        self.assertIsInstance(implicit_mf, ImplicitMF)
        self.assertEqual(expected_features, implicit_mf.features)

    def test_defaultConfigspace_GivenFunctionCalled_correctConfigSpaceReturnedExpected(self):
        cs = ImplicitMF.get_default_configspace()
        params = list(cs.values())
        self.assertIsInstance(cs, CS.ConfigurationSpace)
        self.assertTrue(any(
            param.name == "features" and param.default_value == 1000 and param.lower == 5 and param.upper == 10000 for
            param in params))
        self.assertTrue(any(
            param.name == "ureg" and param.default_value == 0.1 and param.lower == 0.01 and param.upper == 0.1 for param
            in params))
        self.assertTrue(any(
            param.name == "ireg" and param.default_value == 0.1 and param.lower == 0.01 and param.upper == 0.1 for param
            in params))


class TestBiasedMF(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        expected_features = 10
        biased_mf = BiasedMF(expected_features)
        self.assertIsInstance(biased_mf, BiasedMF)
        self.assertEqual(expected_features, biased_mf.features)

    def test_defaultConfigspace_GivenFunctionCalled_correctConfigSpaceReturnedExpected(self):
        cs = BiasedMF.get_default_configspace()
        params = list(cs.values())
        self.assertIsInstance(cs, CS.ConfigurationSpace)
        self.assertTrue(any(
            param.name == "features" and param.default_value == 1000 and param.lower == 2 and param.upper == 10000 for
            param in params))
        self.assertTrue(any(
            param.name == "ureg" and param.default_value == 0.1 and param.lower == 0.01 and param.upper == 0.1 for param
            in params))
        self.assertTrue(any(
            param.name == "ireg" and param.default_value == 0.1 and param.lower == 0.01 and param.upper == 0.1 for param
            in params))
        self.assertTrue(any(
            param.name == "bias" and param.default_value for param in
            params))


if __name__ == '__main__':
    unittest.main()
