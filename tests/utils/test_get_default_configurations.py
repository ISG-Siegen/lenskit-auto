import unittest

from ConfigSpace import Categorical
from ConfigSpace import ConfigurationSpace, Configuration

from lkauto.utils.get_default_configurations import get_default_configurations


class TestGetDefaultConfigurations(unittest.TestCase):

    def setUp(self):
        self.algorithm_list = ['algo1', 'algo2', 'algo3']
        self.config_space = ConfigurationSpace(
            space={
                "algo": Categorical("algo", self.algorithm_list, default="algo1"),
            }
        )
        for algorithm in self.algorithm_list:
            self.config_space.add_configuration_space(
                prefix=algorithm,
                delimiter=":",
                configuration_space=ConfigurationSpace(),
                parent_hyperparameter={"parent": self.config_space["algo"], "value": algorithm},
            )

    def test_getDefaultConfigurations_givenValidConfigurationSpace_correctConfigurationSpaceListReturnedExpected(self):
        result = get_default_configurations(self.config_space)

        # check if the result is a list of configurations with correct length
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        for i, algorithm in enumerate(self.algorithm_list):
            self.assertEqual(algorithm, result[i].get("algo"))

    def test_getDefaultConfigurations_givenConstantAlgorithm_singleConfigurationReturnedExpected(self):
        """Test ConfigSpace with single algorithm (Constant)"""
        from ConfigSpace import Constant

        config_space = ConfigurationSpace(
            space={
                "algo": Constant("algo", "OnlyAlgo"),
            }
        )
        config_space.add_configuration_space(
            prefix="OnlyAlgo",
            delimiter=":",
            configuration_space=ConfigurationSpace(),
            parent_hyperparameter={"parent": config_space["algo"], "value": "OnlyAlgo"},
        )

        result = get_default_configurations(config_space)

        # check if the result is a list of configurations
        self.assertIsInstance(result, list)
        # check if the result list has only 1 config
        self.assertEqual(len(result), 1)
        # check if the single returned config is a Configuration object
        self.assertIsInstance(result[0], Configuration)
        # check if the configuration has the correct algorithm value
        self.assertEqual(result[0].get("algo"), "OnlyAlgo")

    def test_getDefaultConfigurations_givenAlgorithmHyperparameters_correctHyperparameterValuesExpected(self):
        """Test that returned configs include algorithm specific hyperparameters"""
        from ConfigSpace import Integer

        config_space = ConfigurationSpace(
            space={"algo": Categorical("algo", ["algo1"], default="algo1")}
        )

        # Add sub-space with actual hyperparameters
        algo_subspace = ConfigurationSpace()
        algo_subspace.add(Integer("param1", (1, 10), default=5))

        config_space.add_configuration_space(
            prefix="algo1",
            delimiter=":",
            configuration_space=algo_subspace,
            parent_hyperparameter={"parent": config_space["algo"], "value": "algo1"},
        )

        result = get_default_configurations(config_space)

        # check if algorithm parameter is correct
        self.assertEqual(result[0].get("algo"), "algo1")
        # check if hyperparameter is added correctly
        self.assertEqual(result[0].get("algo1:param1"), 5)


if __name__ == '__main__':
    unittest.main()
