import unittest

import ConfigSpace as CS

from lkauto.algorithms.bias import Bias


class TestBias(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        bias = Bias()
        self.assertIsInstance(bias, Bias)

    def test_defaultConfigspace_GivenFunctionCalled_correctConfigSpaceReturnedExpected(self):
        number_item = 10
        number_user = 100
        cs = Bias.get_default_configspace(number_item=number_item, number_user=number_user)
        params = cs.get_hyperparameters()
        self.assertIsInstance(cs, CS.ConfigurationSpace)
        self.assertTrue(any(
            param.name == "item_damping" and param.default_value == 0.025 and param.lower == 1e-4 and param.upper == 850
            for
            param in params))
        self.assertTrue(any(
            param.name == "user_damping" and param.default_value == 0.25 and param.lower == 1e-3 and param.upper == 8500
            for
            param in params))


if __name__ == '__main__':
    unittest.main()
