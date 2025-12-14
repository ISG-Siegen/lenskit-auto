import unittest
from unittest.mock import MagicMock

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