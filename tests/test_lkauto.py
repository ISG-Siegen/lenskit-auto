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
  
    @patch('lkauto.lkauto.build_ensemble')
    @patch('lkauto.lkauto.bayesian_optimization')
    @patch('lkauto.lkauto.preprocess_data')
    def test_getBestPredictionModel_givenEnsembleSizeOne_noEnsembleBuilt(
            self, mock_preprocess, mock_bayesian, mock_ensemble):
        """Test that ensemble is not built when ensemble_size=1:
                if ensemble_size > 1 --> build_ensemble() is called
                else --> incumbent = incumbent.get_dictionary()
        """
        # mock process_data to return the train dataset
        mock_preprocess.return_value = self.train
        # Set up the mock return value for bayesian_optimization
        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'ItemItem'}
        mock_model = MagicMock()
        mock_top_n_runs = pd.DataFrame({'run': [1], 'error': [0.5]})
        mock_bayesian.return_value = (mock_incumbent, mock_model, mock_top_n_runs)

        # Call function with ensemble_size=1
        model, incumbent = get_best_prediction_model(
            train=self.train,
            optimization_strategie='bayesian',
            ensemble_size=1,
            num_evaluations=5,
            save=False
        )
        # check that build_ensemble was not called
        mock_ensemble.assert_not_called()
        
        # check that the model is returned from the optimization and from not ensemble
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_model) 
        
        # check that incumbent is a dict (by checking if get_dictionary() was called in else branch)
        self.assertIsNotNone(incumbent)
        self.assertIsInstance(incumbent, dict)
        self.assertEqual(incumbent, {'algo': 'ItemItem'})
        

    @patch('lkauto.lkauto.bayesian_optimization')
    @patch('lkauto.lkauto.preprocess_data')
    def test_getBestPredictionModel_givenNumEvaluationsNone_setsToInfinity(
            self, mock_preprocess, mock_bayesian):
        """Test that num_evaluations=None is set to np.inf in bayesian_optimization"""
        # setup mocks
        mock_preprocess.return_value = self.train

        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'ItemItem'}
        mock_model = MagicMock()
        mock_top_n_runs = pd.DataFrame({'run': [1], 'error': [0.5]})
        mock_bayesian.return_value = (mock_incumbent, mock_model, mock_top_n_runs)

        # Call function with num_evaluations=None
        model, incumbent = get_best_prediction_model(
            train=self.train,
            optimization_strategie='bayesian',
            num_evaluations=None,
            ensemble_size=1,
            save=False
        )

        # check that bayesian_optimization received np.inf for num_evaluations
        call_kwargs = mock_bayesian.call_args[1] # get the keyword arguments passed to bayesian_optimization
        self.assertEqual(call_kwargs['num_evaluations'], np.inf)


    @patch('lkauto.lkauto.bayesian_optimization')
    @patch('lkauto.lkauto.preprocess_data')
    def test_getBestPredictionModel_givenValidation_splitFoldsSetToOne(
            self, mock_preprocess, mock_bayesian):
        """Test that providing validation overrides split_folds to 1"""
        # Set up mocks
        mock_preprocess.return_value = self.train

        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'ItemItem'}
        mock_model = MagicMock()
        mock_top_n_runs = pd.DataFrame({'run': [1], 'error': [0.5]})
        mock_bayesian.return_value = (mock_incumbent, mock_model, mock_top_n_runs)

        mock_validation = MagicMock()

        # Call with validation + split_folds=5 to hit line 153
        model, incumbent = get_best_prediction_model(
            train=self.train,
            validation=mock_validation,
            optimization_strategie='bayesian',
            split_folds=5,
            num_evaluations=5,
            ensemble_size=1,
            save=False
        )

        # check if split_folds became 1 when validation is provided (if validation is not None)
        call_kwargs = mock_bayesian.call_args[1]
        self.assertEqual(call_kwargs['split_folds'], 1)

    @patch('lkauto.lkauto.Filer')
    @patch('lkauto.lkauto.bayesian_optimization')
    @patch('lkauto.lkauto.preprocess_data')
    def test_getBestPredictionModel_givenSaveTrue_modelAndIncumbentSaved(
            self, mock_preprocess, mock_bayesian, mock_filer_cls):
        """Test that save=True triggers filer.save_model and filer.save_incumbent 
                since save_model() should be called with the model and 
                save_incumbent() should be called with the incumbent dict"""
        # Set up mocks
        mock_preprocess.return_value = self.train

        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'ItemItem'}
        mock_model = MagicMock()
        mock_top_n_runs = pd.DataFrame({'run': [1], 'error': [0.5]})
        mock_bayesian.return_value = (mock_incumbent, mock_model, mock_top_n_runs)

        # Set up the filer mock instance returned by Filer()
        mock_filer = MagicMock()
        mock_filer.output_directory_path = 'output/'
        mock_filer_cls.return_value = mock_filer

        # Call with save=True
        model, incumbent = get_best_prediction_model(
            train=self.train,
            optimization_strategie='bayesian',
            num_evaluations=5,
            ensemble_size=1,
            save=True
        )

        # check if save_model() was called with the model
        mock_filer.save_model.assert_called_once_with(mock_model)
        # check if save_incumbent() was called with the incumbent dict
        mock_filer.save_incumbent.assert_called_once_with(
            mock_incumbent.get_dictionary.return_value)
        
########### TESTS FOR get_best_recommender_model() #############

class TestGetBestRecommenderModel(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal dataset
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [101, 102, 201, 202, 301, 302],
            'rating': [5.0, 4.0, 3.0, 4.0, 2.0, 3.0],
            'timestamp': [1, 2, 3, 4, 5, 6]
        })
        self.train = from_interactions_df(interactions)


    @patch('lkauto.lkauto.bayesian_optimization')
    @patch('lkauto.lkauto.preprocess_data')
    def test_getBestRecommenderModel_givenBayesianStrategy_bayesianOptimizationCalled(
            self, mock_preprocess, mock_bayesian):
        """Test that bayesian_optimization is called for recommender model"""
        # Setup mocks
        mock_preprocess.return_value = self.train
        
        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'ItemItem'}
        mock_model = MagicMock()
        mock_bayesian.return_value = (mock_incumbent, mock_model, None)

        # Call function
        model, incumbent = get_best_recommender_model(
            train=self.train,
            optimization_strategie='bayesian',
            optimization_metric=NDCG,
            num_evaluations=5,
            save=False 
        )

        # check if bayesian_optimization was called
        mock_bayesian.assert_called_once()
        # Check that predict_mode=False was passed 
        # (since it's a recommender model, not a prediction model)
        call_kwargs = mock_bayesian.call_args[1]
        self.assertFalse(call_kwargs['predict_mode'])
        # check if model is returned and not None
        self.assertIsNotNone(model)


    @patch('lkauto.lkauto.random_search')
    @patch('lkauto.lkauto.preprocess_data')
    def test_getBestRecommenderModel_givenRandomSearchStrategy_randomSearchCalled(
            self, mock_preprocess, mock_random):
        """Test that random_search is called for recommender model"""
        # Setup mocks
        mock_preprocess.return_value = self.train
        
        mock_incumbent = MagicMock()
        mock_incumbent.get_dictionary.return_value = {'algo': 'UserUser'}
        mock_model = MagicMock()
        mock_random.return_value = (mock_incumbent, mock_model, None)

        # Call function
        model, incumbent = get_best_recommender_model(
            train=self.train,
            optimization_strategie='random_search',
            optimization_metric=NDCG,
            num_evaluations=5,
            save=False
        )

        # Verify random_search was called
        mock_random.assert_called_once()
        # Check that predict_mode=False was passed
        # (since it's a recommender model, not a prediction model)
        call_kwargs = mock_random.call_args[1]
        self.assertFalse(call_kwargs['predict_mode'])
        # check if model is returned and not None
        self.assertIsNotNone(model)


