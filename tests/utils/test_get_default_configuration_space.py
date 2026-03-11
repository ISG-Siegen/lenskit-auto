import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace
from lenskit.data import from_interactions_df

from lkauto.utils.get_default_configuration_space import get_default_configuration_space
from lkauto.utils.validation_split import validation_split


class TestGetDefaultConfigurationSpace(unittest.TestCase):
    model_mock = MagicMock()
    model_mock.get_default_configspace.return_value = ConfigurationSpace()

    def setUp(self):
        self.random_state = 42
        self.df = pd.DataFrame(np.array([[1, 1, 1],
                                         [1, 2, 2],
                                         [1, 3, 3],
                                         [2, 1, 1],
                                         [2, 2, 2],
                                         [2, 3, 3],
                                         [3, 1, 1],
                                         [3, 2, 2],
                                         [3, 3, 3],
                                         [4, 1, 1],
                                         [4, 2, 2],
                                         [4, 3, 3],
                                         [5, 1, 1],
                                         [5, 2, 2],
                                         [5, 3, 3]]), columns=["user", "item", "rating", ])
        # self.df = pd.DataFrame(np.ones((100, 3), dtype=int), columns=["user", "item", "rating", ])
        self.ds = from_interactions_df(self.df, user_col="user", item_col="item", rating_col="rating")
        self.val_fold_indices = validation_split(data=self.ds, random_state=42)

    def test_getDefaultConfigurationSpace_givenInvalidFeedback_valueErrorThrown(self):
        """Test that ValueError is raised for unknown feedback type"""
        with self.assertRaises(ValueError) as cm:
            get_default_configuration_space(feedback="", data=self.ds,
                                            val_fold_indices=self.val_fold_indices,
                                            random_state=self.random_state)
        self.assertEqual("Unknown feedback type: ", cm.exception.args[0])

    @patch('lkauto.utils.get_default_configuration_space.ItemItem', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.UserUser', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.ImplicitMF', model_mock)
    def test_getDefaultConfigurationSpace_GivenImplicitAndValidInputs_CorrectConfigSpaceReturnedExpected(self):
        """Test that valid inputs for implicit feedback return correct ConfigurationSpace"""
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
        """Test that valid inputs for explicit feedback return correct ConfigurationSpace"""
        algorithm_list_string = 'ItemItem, UserUser, FunkSVD, BiasedSVD, ALSBiasedMF, Bias'

        result = get_default_configuration_space(feedback='explicit', data=self.ds,
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
    def test_getDefaultConfigurationSpace_GivenTTSplitIteratorAsInput_CorrectConfigSpaceReturnedExpected(self):
        """Test that passing an TTSplit-Iterator as data works correctly"""
        # Create new folds specifically for this test
        folds = validation_split(self.ds, random_state=42)

        result = get_default_configuration_space(feedback='explicit',
                                                 data=folds,
                                                 random_state=self.random_state)

        self.assertIsInstance(result, ConfigurationSpace)

    @patch('lkauto.utils.get_default_configuration_space.ItemItem')
    def test_getDefaultConfigurationSpace_GivenCorrectDatasetSizeToAlgorithms(self, mock_itemitem):
        """Test that algorithms receive correct num_users and num_items"""
        mock_itemitem.get_default_configspace.return_value = ConfigurationSpace()

        result = get_default_configuration_space(feedback='implicit', data=self.ds, random_state=42)

        # Check ItemItem.get_default_configspace was called with correct parameters
        # check if get_default_configspace() was called at least once
        mock_itemitem.get_default_configspace.assert_called()
        call_params = mock_itemitem.get_default_configspace.call_args[1]
        self.assertEqual(call_params['number_user'], 5)
        self.assertEqual(call_params['number_item'], 3)
        self.assertIsInstance(result, ConfigurationSpace)

    @patch('lkauto.utils.get_default_configuration_space.ItemItem', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.UserUser', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.FunkSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedMF', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.Bias', model_mock)
    def test_getDefaultConfigurationSpace_GivenDataWithSingleUser_CorrectConfigSpaceReturnedExpected(self):
        """Test dataset with only 1 user"""
        df = pd.DataFrame([[1, 1, 5], [1, 2, 4]], columns=["user", "item", "rating"])
        ds = from_interactions_df(df, user_col="user", item_col="item", rating_col="rating")

        result = get_default_configuration_space(feedback='explicit', data=ds, random_state=42)

        self.assertIsInstance(result, ConfigurationSpace)

    @patch('lkauto.utils.get_default_configuration_space.ItemItem', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.UserUser', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.FunkSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedMF', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.Bias', model_mock)
    def test_getDefaultConfigurationSpace_GivenDataWithSingleItem_CorrectConfigSpaceReturnedExpected(self):
        """Test dataset with only 1 item"""
        df = pd.DataFrame([[1, 1, 5], [2, 1, 4], [3, 1, 3]], columns=["user", "item", "rating"])
        ds = from_interactions_df(df, user_col="user", item_col="item", rating_col="rating")

        result = get_default_configuration_space(feedback='explicit', data=ds, random_state=42)

        self.assertIsInstance(result, ConfigurationSpace)

    def test_getDefaultConfigurationSpace_GivenMultipleFolds_usesMinimumSizeExpected(self):
        """Test that with TTSplit-Iterator, minimum user and item counts are used"""
        # Create mock folds with know different sizes
        # Fold 1: 10 users, 8 items
        mock_fold1 = MagicMock()
        mock_fold1.train.user_count = 10
        mock_fold1.train.item_count = 8

        # Fold 2: 12 users, 6 items
        mock_fold2 = MagicMock()
        mock_fold2.train.user_count = 12
        mock_fold2.train.item_count = 6

        # Fold 3: 8 users, 10 items
        mock_fold3 = MagicMock()
        mock_fold3.train.user_count = 8
        mock_fold3.train.item_count = 10

        # concatenate folds into a list and create an iterator
        folds_list = [mock_fold1, mock_fold2, mock_fold3]
        folds_iterator = iter(folds_list)

        # Test
        with patch('lkauto.utils.get_default_configuration_space.ItemItem') as mock:
            mock.get_default_configspace.return_value = ConfigurationSpace()
            result = get_default_configuration_space(
                feedback='implicit',
                data=folds_iterator,
                random_state=42
            )

            call_params = mock.get_default_configspace.call_args[1]
            self.assertEqual(call_params['number_user'], 8)
            self.assertEqual(call_params['number_item'], 6)
            self.assertIsInstance(result, ConfigurationSpace)


if __name__ == '__main__':
    unittest.main()
