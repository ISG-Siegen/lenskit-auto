import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace
from lenskit.data import from_interactions_df, DatasetBuilder

from lkauto.utils.get_default_configuration_space import get_default_configuration_space
from lkauto.utils.validation_split import validation_split


class TestGetDefaultConfigurationSpace(unittest.TestCase):
    model_mock = MagicMock()
    model_mock.get_default_configspace.return_value = ConfigurationSpace()

    def setUp(self):
        self.random_state = 42
        self.df = pd.DataFrame(np.array([[1,1,1],
                            [1,2,2],
                            [1,3,3],
                            [2,1,1],
                            [2,2,2],
                            [2,3,3],
                            [3,1,1],
                            [3,2,2],
                            [3,3,3],
                            [4,1,1],
                            [4,2,2],
                            [4,3,3],
                            [5,1,1],
                            [5,2,2],
                            [5,3,3]]), columns=["user", "item", "rating", ])
        # self.df = pd.DataFrame(np.ones((100, 3), dtype=int), columns=["user", "item", "rating", ])
        self.ds = from_interactions_df(self.df, user_col="user", item_col="item", rating_col="rating")
        self.val_fold_indices = validation_split(data=self.ds, random_state=42)

    def test_getDefaultConfigurationSpace_givenInvalidFeedback_valueErrorThrown(self):
        with self.assertRaises(ValueError) as cm:
            get_default_configuration_space(feedback="", data=self.ds,
                                            val_fold_indices=self.val_fold_indices,
                                            random_state=self.random_state)
        self.assertEqual("Unknown feedback type: ", cm.exception.args[0])

    @patch('lkauto.utils.get_default_configuration_space.ItemItem', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.UserUser', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.FunkSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.ImplicitMF', model_mock)
    def test_getDefaultConfigurationSpace_GivenImplicitAndValidInputs_CorrectConfigSpaceReturnedExpected(self):
        algorithm_list_string = 'ItemItem, UserUser, ImplicitMF'

        result = get_default_configuration_space(feedback='implicit', data=self.ds,
                                                 val_fold_indices=self.val_fold_indices,
                                                 random_state=self.random_state)

        self.assertIsInstance(result, ConfigurationSpace)
        self.assertTrue(algorithm_list_string in str(result.get('algo')))

    @patch('lkauto.utils.get_default_configuration_space.ItemItem', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.UserUser', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.FunkSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedMF', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.Bias', model_mock)
    def test_getDefaultConfigurationSpace_GivenExplicitAndValidInputs_CorrectConfigSpaceReturnedExpected(self):
        algorithm_list_string = 'ItemItem, UserUser, FunkSVD, BiasedSVD, ALSBiasedMF, Bias'

        result = get_default_configuration_space(feedback='explicit', data=self.ds,
                                                 val_fold_indices=self.val_fold_indices,
                                                 random_state=self.random_state)

        self.assertIsInstance(result, ConfigurationSpace)
        self.assertTrue(algorithm_list_string in str(result.get('algo')))


if __name__ == '__main__':
    unittest.main()
