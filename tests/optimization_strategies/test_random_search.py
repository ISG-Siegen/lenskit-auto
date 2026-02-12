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
            cs=None,  # Passing None instead of a ConfigurationSpace
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

    @patch('lkauto.optimization_strategies.random_search.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.ImplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.get_default_configurations')
    def test_randomSearch_givenImplicit_returnsNoneTopN(self, mock_get_defaults, mock_evaler, mock_explicit_evaler):
        """Test that implicit feedback returns None for top_n_runs"""
        # Setup mocks
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance
        mock_evaler_instance.evaluate.return_value = (0.7, MagicMock())

        # mock for implicit evaler
        config1 = MagicMock()
        mock_get_defaults.return_value = [config1]

        mock_cs = MagicMock()

        best_config, best_model, top_n = random_search(
            train=self.train,
            user_feedback='implicit',
            validation=self.validation,
            cs=mock_cs,
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            num_evaluations=1,
            minimize_error_metric_val=True,
            random_state=42
        )

        # check that top_n is None for implicit feedback
        self.assertIsNone(top_n)

    @patch('lkauto.optimization_strategies.random_search.time')
    @patch('lkauto.optimization_strategies.random_search.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.get_default_configurations')
    def test_randomSearch_givenTimeBased_loopsUntilTimeLimit(self, mock_get_defaults, mock_evaler, mock_time):
        """Test time-based search (num_evaluations=0) with both minimize branches"""
        # Test both minimize=True and minimize=False to cover both branches
        for minimize in [True, False]:
            with self.subTest(minimize=minimize):
                # reset mocks between test iterations
                mock_evaler.reset_mock()
                mock_get_defaults.reset_mock()
                mock_time.reset_mock()

                # create mock evaler instance that will be returned when ExplicitEvaler() is called
                mock_evaler_instance = MagicMock()
                mock_evaler.return_value = mock_evaler_instance
                # mock evaluate to return error=0.8
                mock_evaler_instance.evaluate.return_value = (0.8, MagicMock())
                # mock top_n_runs attribute needed for explicit feedback return value
                mock_evaler_instance.top_n_runs = pd.DataFrame()

                # mock get_default_configurations to return a list with one default config
                default_config = (("algo", "ItemItem"), ("param", 1))
                mock_get_defaults.return_value = [default_config]

                # create mock ConfigurationSpace
                mock_cs = MagicMock()
                # use another tuple for sampled configurations
                sampled_config = (("algo", "UserUser"), ("param", 2))
                mock_cs.sample_configuration.return_value = sampled_config

                # Mock time.time() to control when loop exits
                # returns: 0 (start_time), then 0.1, 0.2, 0.3 ... then 100 (loop exits)
                mock_time.time.side_effect = [0, 0.1, 0.2, 0.3, 100]

                # Call random_search in time-based mode (num_evaluations=0)
                best_config, best_model, top_n = random_search(
                    train=self.train,
                    user_feedback='explicit',
                    validation=self.validation,
                    cs=mock_cs,
                    optimization_metric=self.optimization_metric,
                    filer=self.filer,
                    num_evaluations=0,
                    time_limit_in_sec=1,
                    minimize_error_metric_val=minimize,  # Test both True and False branches
                    random_state=42
                )

                # check that at least one evaluation happened
                self.assertGreater(mock_evaler_instance.evaluate.call_count, 0)
                # check that a best config was selected
                self.assertIsNotNone(best_config)

    @patch('lkauto.optimization_strategies.random_search.time')
    @patch('lkauto.optimization_strategies.random_search.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.random_search.get_default_configurations')
    def test_randomSearch_givenNumEvalsWithTimeLimit_stopsEarly(self, mock_get_defaults, mock_evaler, mock_time):
        """Test line 209: time limit break inside num_evaluations loop"""
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance
        mock_evaler_instance.evaluate.return_value = (0.5, MagicMock())
        mock_evaler_instance.top_n_runs = pd.DataFrame()

        # mock get_default_configurations to return a list with two configs (for multiple evaluations)
        config1 = (("algo", "ItemItem"), ("id", 1))
        config2 = (("algo", "UserUser"), ("id", 2))
        mock_get_defaults.return_value = [config1, config2]
        # mock configuration space
        mock_cs = MagicMock()
        mock_cs.sample_configuration.return_value = config1

        # start_time=0, after the first eval time=100 (exceeds limit of 1)
        mock_time.time.side_effect = [0, 100]

        best_config, best_model, top_n = random_search(
            train=self.train,
            user_feedback='explicit',
            validation=self.validation,
            cs=mock_cs,
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            num_evaluations=2,
            time_limit_in_sec=1,
            random_state=42
        )

        # only 1 eval should have run before time limit started
        self.assertEqual(mock_evaler_instance.evaluate.call_count, 1)
