import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from ConfigSpace import ConfigurationSpace, Categorical

from lkauto.optimization_strategies.random_search import random_search


class TestRandomSearch(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Create mock train dataset
        self.train = MagicMock()
        self.train.__class__.__name__ = 'Dataset'

        # Create mock validation
        self.validation = MagicMock()
        self.validation.__class__.__name__ = 'ItemListCollection'

        # Create a simple ConfigurationSpace
        self.cs = ConfigurationSpace(
            space={"algo": Categorical("algo", ["ItemItem", "UserUser"], default="ItemItem")}
        )

        # Create mock filer
        self.filer = MagicMock()

        # Create mock optimization metric
        self.optimization_metric = MagicMock()

    def test_randomSearch_givenInvalidFeedback_valueErrorThrown(self):
        """Test that ValueError is raised for invalid feedback types"""
        invalid_feedbacks = ["", "IMPLICIT", "EXPLICIT", None, 12345, "both", "random"]

        for invalid_feedback in invalid_feedbacks:
            with self.subTest(feedback=invalid_feedback):
                with self.assertRaises(ValueError) as cm:
                    random_search(
                        train=self.train,
                        user_feedback=invalid_feedback,
                        validation=self.validation,
                        cs=self.cs,
                        optimization_metric=self.optimization_metric,
                        filer=self.filer,
                        num_evaluations=5
                    )
                self.assertIn("feedback must be either explicit or implicit", str(cm.exception))

    @patch('lkauto.optimization_strategies.random_search.ImplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.get_default_configurations')
    def test_randomSearch_givenExplicitFeedback_explicitEvalerCreated(self, mock_get_defaults, mock_evaler, mock_implicit_evaler):
        """Test that ExplicitEvaler is initialized for explicit feedback"""
        # Setup mocks
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance
        mock_evaler_instance.evaluate.return_value = (0.9, MagicMock())
        mock_evaler_instance.top_n_runs = pd.DataFrame()
        mock_get_defaults.return_value = []

        # Call function
        random_search(
            train=self.train,
            user_feedback='explicit',
            validation=self.validation,
            cs=self.cs,
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            num_evaluations=1,
            random_state=42
        )

        # Verify ExplicitEvaler was called
        mock_evaler.assert_called_once()
        # extract call arguments to verify
        call_kwargs = mock_evaler.call_args[1]
        # verify that correct arguments are passed
        self.assertEqual(call_kwargs['train'], self.train)
        self.assertEqual(call_kwargs['validation'], self.validation)
        self.assertEqual(call_kwargs['optimization_metric'], self.optimization_metric)

    @patch('lkauto.optimization_strategies.random_search.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.ImplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.get_default_configurations')
    def test_randomSearch_givenImplicitFeedback_implicitEvalerCreated(self, mock_get_defaults, mock_evaler, mock_explicit_evaler):
        """Test that ImplicitEvaler is initialized for implicit feedback"""
        # Setup mocks
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance
        mock_evaler_instance.evaluate.return_value = (0.9, MagicMock())
        mock_get_defaults.return_value = []

        # Call function
        random_search(
            train=self.train,
            user_feedback='implicit',
            validation=self.validation,
            cs=self.cs,
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            num_evaluations=1,
            random_state=42
        )

        # Verify ImplicitEvaler was called
        mock_evaler.assert_called_once()
        # extract call arguments to verify
        call_kwargs = mock_evaler.call_args[1]
        # verify that correct arguments are passed
        self.assertEqual(call_kwargs['train'], self.train)
        self.assertEqual(call_kwargs['validation'], self.validation)
        self.assertEqual(call_kwargs['optimization_metric'], self.optimization_metric)  
      
    @patch('lkauto.optimization_strategies.random_search.ImplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.get_default_configuration_space')
    @patch('lkauto.optimization_strategies.random_search.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.get_default_configurations')
    def test_randomSearch_givenNoConfigSpace_defaultConfigSpaceCreated(self, mock_get_defaults, mock_evaler, mock_get_cs, mock_implicit_evaler):
        """Test that default ConfigurationSpace is created when cs=None"""
        # Setup mocks
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance
        mock_evaler_instance.evaluate.return_value = (0.5, MagicMock())
        mock_evaler_instance.top_n_runs = pd.DataFrame()
        mock_evaler_instance.train_test_splits = []

        default_cs = ConfigurationSpace(
            space={"algo": Categorical("algo", ["UserUser"], default="UserUser")}
        )
        mock_get_cs.return_value = default_cs
        mock_get_defaults.return_value = []

        # Call function with cs=None
        random_search(
            train=self.train,
            user_feedback='explicit',
            validation=self.validation,
            cs=None, # Passing None instead of a ConfigurationSpace
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            num_evaluations=1,
            random_state=42
        )

        # verify get_default_configuration_space was called
        mock_get_cs.assert_called_once()
        # get the call arguments to verify
        call_args = mock_get_cs.call_args
        # verify that correct arguments are passed
        self.assertEqual(call_args[1]['data'], self.train)
        self.assertEqual(call_args[1]['val_fold_indices'], [])
        self.assertEqual(call_args[1]['feedback'], 'explicit')  

