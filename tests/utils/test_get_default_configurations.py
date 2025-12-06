import unittest

from ConfigSpace import Categorical
from ConfigSpace import ConfigurationSpace

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


if __name__ == '__main__':
    unittest.main()
