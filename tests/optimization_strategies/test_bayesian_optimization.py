import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from ConfigSpace import ConfigurationSpace, Categorical

from lkauto.optimization_strategies.bayesian_optimization import bayesian_optimization




class TestBayesianOptimization(unittest.TestCase):

    def setUp(self):
        """Set up common test fixtures"""
        # Create mock train dataset
        self.train = MagicMock()

        # Create mock validation
        self.validation = MagicMock()

        # Create simple ConfigurationSpace
        self.cs = ConfigurationSpace(
            space={"algo": Categorical("algo", ["ItemItem"], default="ItemItem")}
        )

        # Create mock filer
        self.filer = MagicMock()
        self.filer.get_smac_output_directory_path.return_value = '/tmp/smac_output'

        # Mock optimization metric
        self.optimization_metric = MagicMock()


    def test_bayesianOptimization_givenInvalidFeedback_valueErrorThrown(self):
        """Test that ValueError is raised for invalid feedback types"""
        invalid_feedbacks = ["", "IMPLICIT", "EXPLICIT", None, 12345, "both", "random"]

        for invalid_feedback in invalid_feedbacks:
            with self.subTest(feedback=invalid_feedback):
                with self.assertRaises(ValueError) as cm:
                    bayesian_optimization(
                        train=self.train,
                        user_feedback=invalid_feedback,
                        validation=self.validation,
                        cs=self.cs,
                        optimization_metric=self.optimization_metric,
                        filer=self.filer
                    )
                self.assertIn("feedback must be either explicit or implicit", str(cm.exception))  

    @patch('lkauto.optimization_strategies.bayesian_optimization.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.bayesian_optimization.HyperparameterOptimizationFacade')
    def test_bayesianOptimization_givenExplicitFeedback_explicitEvalerCreated(self, mock_smac, mock_evaler):
        """Test that ExplicitEvaler is created when the feedback is explicit"""
        # Setup mocks
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance
        mock_evaler_instance.top_n_runs = pd.DataFrame()

        mock_smac_instance = MagicMock()
        mock_smac.return_value = mock_smac_instance

        # run the bayesian optimization function
        bayesian_optimization(
            train=self.train,
            user_feedback='explicit',
            validation=self.validation,
            cs=self.cs,
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            random_state=42
        )
        # Verify ExplicitEvaler was called
        mock_evaler.assert_called_once()

    @patch('lkauto.optimization_strategies.bayesian_optimization.ImplicitEvaler')
    @patch('lkauto.optimization_strategies.bayesian_optimization.HyperparameterOptimizationFacade')
    def test_bayesianOptimization_givenImplicitFeedback_implicitEvalerCreated(self, mock_smac, mock_evaler):
        """Test that ImplicitEvaler is created for implicit feedback"""
        # Setup mocks
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance

        mock_smac_instance = MagicMock()
        mock_smac.return_value = mock_smac_instance
    
        # Call function
        bayesian_optimization(
            train=self.train,
            user_feedback='implicit',
            validation=self.validation,
            cs=self.cs,
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            random_state=42
        )

        # Verify ExplicitEvaler was called
        mock_evaler.assert_called_once()

    @patch('lkauto.optimization_strategies.bayesian_optimization.ImplicitEvaler')
    @patch('lkauto.optimization_strategies.bayesian_optimization.get_default_configuration_space')
    @patch('lkauto.optimization_strategies.bayesian_optimization.ExplicitEvaler')
    @patch('lkauto.optimization_strategies.bayesian_optimization.HyperparameterOptimizationFacade')
    def test_bayesianOptimization_givenNoConfigSpace_defaultConfigSpaceCreated(self, mock_smac, mock_evaler, mock_get_cs, mock_implicit_evaler):
        """Test that default ConfigurationSpace is created when cs=None"""
        # Setup mocks
        mock_evaler_instance = MagicMock()
        mock_evaler.return_value = mock_evaler_instance
        mock_evaler_instance.top_n_runs = pd.DataFrame()

        default_cs = ConfigurationSpace(
            space={"algo": Categorical("algo", ["UserUser"], default="UserUser")}
        )
        mock_get_cs.return_value = default_cs

        mock_smac_instance = MagicMock()
        mock_smac.return_value = mock_smac_instance

        # Call function with cs=None
        bayesian_optimization(
            train=self.train,
            user_feedback='explicit',
            validation=self.validation,
            cs=None,  # Passing None instead of a ConfigurationSpace
            optimization_metric=self.optimization_metric,
            filer=self.filer,
            random_state=42
        )

        # verify get_default_configuration_space was called
        mock_get_cs.assert_called_once()
        # get the call arguments to verify
        call_kwargs = mock_get_cs.call_args[1]
        # verify that correct arguments are passed
        self.assertEqual(call_kwargs['data'], self.train)
        self.assertEqual(call_kwargs['validation'], self.validation)
        self.assertEqual(call_kwargs['feedback'], 'explicit')