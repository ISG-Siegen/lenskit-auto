import unittest

import ConfigSpace as CS

from lkauto.algorithms.item_knn import ItemItem


class TestItemKNN(unittest.TestCase):
    def test_init_givenObjectInitialized_ObjectInitializedCorrectlyExpected(self):
        nnbrs = 10
        item_item = ItemItem(nnbrs)
        self.assertEqual(nnbrs, item_item.nnbrs)
        self.assertIsInstance(item_item, ItemItem)

    def test_defaultConfigspace_GivenFunctionCalled_correctConfigSpaceReturnedExpected(self):
        cs = ItemItem.get_default_configspace()
        params = cs.get_hyperparameters()
        self.assertIsInstance(cs, CS.ConfigurationSpace)
        self.assertTrue(any(
            param.name == "nnbrs" and param.default_value == 1000 and param.lower == 1 and param.upper == 10000 for
            param in params))
        self.assertTrue(any(
            param.name == "min_nbrs" and param.default_value == 1 and param.lower == 1 and param.upper == 1000 for
            param in params))
        self.assertTrue(any(
            param.name == "min_sim" and param.default_value == 1.0e-6 and param.lower == 1.0e-10 and param.upper ==
            1.0e-2 for param in params))


if __name__ == '__main__':
    unittest.main()
