import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from lenskit.metrics import RMSE, NDCG
from lenskit.data import from_interactions_df

from lkauto.lkauto import get_best_prediction_model, get_best_recommender_model


class TestGetBestPredictionModel(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal dataset with 3 users and 2 ratings each
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [101, 102, 201, 202, 301, 302],
            'rating': [5.0, 4.0, 3.0, 4.0, 2.0, 3.0],
            'timestamp': [1, 2, 3, 4, 5, 6]
        })
        self.train = from_interactions_df(interactions)

    @patch('lkauto.lkauto.bayesian_optimization')
    @patch('lkauto.lkauto.preprocess_data')
    @patch('lkauto.lkauto.build_ensemble')
    def test_getBestPredictionModel_givenBayesianStrategy_bayesianOptimizationCalled(
            self, mock_ensemble, mock_preprocess, mock_bayesian):
        """Test that bayesian_optimization is called with bayesian strategy"""
        # mock preprocess_data to return the train dataset
        mock_preprocess.return_value = self.train
        
        # Set up the mock return value for bayesian_optimization
        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'ItemItem'}
        mock_model = MagicMock()
        mock_top_n_runs = pd.DataFrame({'run': [1], 'error': [0.5]})
        mock_bayesian.return_value = (mock_incumbent, mock_model, mock_top_n_runs)
        # Set up the mock return value for build_ensemble
        mock_ensemble.return_value = (mock_model, mock_incumbent.get_dictionary())

        # Call function
        model, incumbent = get_best_prediction_model(
            train=self.train,
            optimization_strategie='bayesian',
            optimization_metric=RMSE,
            ensemble_size=2, # > 1 to trigger build_ensemble
            num_evaluations=5,
            save=False
        )

        # verify bayesian_optimization was called
        mock_bayesian.assert_called_once()
        # check if model is returned and not None
        self.assertIsNotNone(model)
        # check if incumbent is returned and not None
        self.assertIsNotNone(incumbent)

    @patch('lkauto.lkauto.random_search')
    @patch('lkauto.lkauto.preprocess_data')
    @patch('lkauto.lkauto.build_ensemble')
    def test_getBestPredictionModel_givenRandomSearchStrategy_randomSearchCalled(
            self, mock_ensemble, mock_preprocess, mock_random):
        """Test that random_search is called with random_search strategy"""
        # mock preprocess_data to return the train dataset
        mock_preprocess.return_value = self.train
        
        # Set up the mock return value for random_search
        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'UserUser'}
        mock_model = MagicMock()
        mock_top_n_runs = pd.DataFrame({'run': [1], 'error': [0.5]})
        mock_random.return_value = (mock_incumbent, mock_model, mock_top_n_runs)
        # Set up the mock return value for build_ensemble
        mock_ensemble.return_value = (mock_model, mock_incumbent.get_dictionary())

        # Call function
        model, incumbent = get_best_prediction_model(
            train=self.train,
            optimization_strategie='random_search',
            optimization_metric=RMSE,
            ensemble_size=2,
            num_evaluations=5,
            save=False
        )

        # verify random_search was called
        mock_random.assert_called_once()
        # check if model is returned and not None
        self.assertIsNotNone(model)
        # check if incumbent is returned and not None
        self.assertIsNotNone(incumbent)

    @patch('lkauto.lkauto.preprocess_data')
    def test_getBestPredictionModel_givenInvalidStrategy_valueErrorRaised(self, mock_preprocess):
        """Test that ValueError is raised for giving an invalid optimization strategy"""
        # mock preprocess_data to return the train dataset
        mock_preprocess.return_value = self.train

        with self.assertRaises(ValueError) as cm:
            get_best_prediction_model(
                train=self.train,
                optimization_strategie='optimize', # invalid value
                num_evaluations=5
            )
        # check thta the error message is the same as the expected one in the function
        self.assertIn('optimization_strategie must be either bayesian or random_search',
                     str(cm.exception))
  