import unittest
from unittest.mock import MagicMock

from ConfigSpace import ConfigurationSpace, Categorical, Integer

from lenskit.als import ImplicitMFScorer, BiasedMFScorer
from lenskit.basic import BiasScorer
from lenskit.funksvd import FunkSVDScorer
from lenskit.knn import UserKNNScorer, ItemKNNScorer
from lenskit.sklearn.svd import BiasedSVDScorer

from lkauto.utils.get_model_from_cs import get_model_from_cs


class TestGetModelFromCS(unittest.TestCase):

    def setUp(self):
        self.cs = MagicMock()
        self.random_state = 42

    def test_getModelFromCS_givenInvalidFeedback_valueErrorThrown(self):
        """Test that ValueError is raised for various invalid feedback types"""
        invalid_feedbacks_list = ["", "IMPLICIT", "EXPLICIT", None, 12345, "both"]
        for invalid_feedback in invalid_feedbacks_list:
            with self.subTest(feedback=invalid_feedback):
                with self.assertRaises(ValueError) as cm:
                    get_model_from_cs(cs=self.cs, random_state=42,
                                      feedback=invalid_feedback)
                self.assertIn("Unknown feedback type", str(cm.exception))

    def test_getModelFromCS_givenInvalidAlgorithmInCS_valueErrorThrown(self):
        """ Test that ValueError is raised for unknown algorithm in Configuration"""
        self.cs.get.return_value = 'alg'
        self.cs.items.return_value = [('algo', 'alg'), ('alg:attr1', 'val1'), ('alg:attr2', 'val2')]

        with self.assertRaises(ValueError) as cm:
            get_model_from_cs(cs=self.cs, random_state=self.random_state,
                              feedback="implicit")
        self.assertEqual("Unknown algorithm: alg", cm.exception.args[0])

    def test_getModelFromCS_givenImplicitAndValidInputs_correctModelReturnedExpected(self):
        """Test each implicit algorithm individually to ensure correct type is returned"""
        algorithm_params_list = [
            ('ItemItem', [('algo', 'ItemItem')], ItemKNNScorer),
            ('UserUser', [('algo', 'UserUser')], UserKNNScorer),
            ('ImplicitMF', [('algo', 'ImplicitMF')], ImplicitMFScorer)]

        for algorithm_params in algorithm_params_list:
            with self.subTest(algorithm=algorithm_params[0]):
                # mock setup
                self.cs.get.return_value = algorithm_params[0]
                self.cs.items.return_value = algorithm_params[1]

                result = get_model_from_cs(cs=self.cs, random_state=self.random_state, feedback="implicit")

                # check correct model type returned
                self.assertIsInstance(result, algorithm_params[-1])

    def test_getModelFromCS_givenExplicitAndValidInputs_correctModelReturnedExpected(self):
        """Test each explicit algorithm individually to ensure correct type is returned"""
        # algorithm_name, mock_items_return_value, expected_model_class
        algorithm_params_list = [
            ('ItemItem', [('algo', 'ItemItem')], ItemKNNScorer),
            ('UserUser', [('algo', 'UserUser')], UserKNNScorer),
            ('FunkSVD', [('algo', 'FunkSVD')], FunkSVDScorer),
            ('BiasedSVD', [('algo', 'BiasedSVD')], BiasedSVDScorer),
            ('ALSBiasedMF', [('algo', 'ALSBiasedMF')], BiasedMFScorer),
            ('Bias', [('algo', 'Bias')], BiasScorer)
        ]

        for algorithm_params in algorithm_params_list:
            with self.subTest(algorithm=algorithm_params[0]):
                # mock setup
                self.cs.get.return_value = algorithm_params[0]
                self.cs.items.return_value = algorithm_params[1]

                result = get_model_from_cs(cs=self.cs, random_state=self.random_state, feedback="explicit")

                # check correct model type returned
                self.assertIsInstance(result, algorithm_params[-1])

    def test_getModelFromCS_givenRealConfigurationImplicit_correctModelReturnedExpected(self):
        """Test that real Configuration works for implicit algorithms and check
            if feedback is set correctly"""
        test_cases = [
            ('ItemItem', ItemKNNScorer),
            ('UserUser', UserKNNScorer),
            ('ImplicitMF', ImplicitMFScorer),
        ]

        for algo_name, expected_type in test_cases:
            with self.subTest(algorithm=algo_name):
                config_space = ConfigurationSpace(
                    space={"algo": Categorical("algo", [algo_name], default=algo_name)}
                )
                config = config_space.get_default_configuration()

                result = get_model_from_cs(cs=config, feedback='implicit', random_state=self.random_state)

                # check correct model type returned
                self.assertIsInstance(result, expected_type)

                # check feedback is set correctly (if model supports it)
                if hasattr(result.config, 'feedback'):
                    self.assertEqual(result.config.feedback, 'implicit')

    def test_getModelFromCS_givenRealConfigurationExplicit_correctModelReturnedExpected(self):
        """Test that real Configuration works for explicit algorithms and check
             if feedback is set correctly"""
        test_cases = [
            ('ItemItem', ItemKNNScorer),
            ('UserUser', UserKNNScorer),
            ('FunkSVD', FunkSVDScorer),
            ('BiasedSVD', BiasedSVDScorer),
            ('ALSBiasedMF', BiasedMFScorer),
            ('Bias', BiasScorer),
        ]

        for algo_name, expected_type in test_cases:
            with self.subTest(algorithm=algo_name):
                config_space = ConfigurationSpace(
                    space={"algo": Categorical("algo", [algo_name], default=algo_name)}
                )
                config = config_space.get_default_configuration()
                result = get_model_from_cs(cs=config, feedback='explicit', random_state=self.random_state)

                # check correct model type returned
                self.assertIsInstance(result, expected_type)

                # check feedback is set correctly (if model supports it)
                if hasattr(result.config, 'feedback'):
                    self.assertEqual(result.config.feedback, 'explicit')

    def test_getModelFromCS_givenHyperparameters_correctDefaultHyperparametersExpected(self):
        """Test that hyperparameters from Configuration are correctly used and applied to the model"""
        # create Configuration with custom hyperparameters
        config_space = ConfigurationSpace(
            space={
                "algo": Categorical("algo", ["ItemItem"], default="ItemItem"),
            }
        )
        config_space.add(Integer("ItemItem:min_nbrs", (1, 10), default=5))
        config_space.add(Integer("ItemItem:max_nbrs", (10, 50), default=30))

        config = config_space.get_default_configuration()

        result = get_model_from_cs(cs=config, feedback='implicit', random_state=self.random_state)

        # check hyperparameters are correctly applied
        self.assertEqual(result.config.min_nbrs, 5)
        self.assertEqual(result.config.max_nbrs, 30)


if __name__ == '__main__':
    unittest.main()
