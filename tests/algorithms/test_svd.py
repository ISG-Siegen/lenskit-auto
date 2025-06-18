import unittest

import ConfigSpace as CS

from lkauto.algorithms.svd import BiasedSVD


class TestSVD(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        expected_features = 10
        biased_svd = BiasedSVD(expected_features)
        self.assertIsInstance(biased_svd, BiasedSVD)
        self.assertEqual(expected_features, biased_svd.features)

    def test_defaultConfigspace_GivenFunctionCalled_correctConfigSpaceReturnedExpected(self):
        number_item_list = [1500, 100000]
        for number_item in number_item_list:
            cs = BiasedSVD.get_default_configspace(number_item=number_item)
            params = list(cs.values())
            with self.subTest(number_item=number_item):
                self.assertIsInstance(cs, CS.ConfigurationSpace)
                if number_item < 10000:
                    self.assertTrue(any(
                        param.name == "features" and param.default_value == 1499 and param.lower == 2
                        and param.upper == 1500 for param in params))
                else:
                    self.assertTrue(any(
                        param.name == "features" and param.default_value == 1000 and param.lower == 2
                        and param.upper == 10000 for param in params))
                self.assertTrue(any(
                    param.name == "damping" and param.default_value == 5 and param.lower == 0.0001
                    and param.upper == 1000 for param in params))


if __name__ == '__main__':
    unittest.main()
