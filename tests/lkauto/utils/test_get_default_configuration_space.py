import unittest
from unittest.mock import patch, MagicMock

from ConfigSpace import ConfigurationSpace

from lkauto.utils.get_default_configuration_space import get_default_configuration_space


class TestGetDefaultConfigurationSpace(unittest.TestCase):
    model_mock = MagicMock()
    model_mock.get_default_configspace.return_value = ConfigurationSpace()

    def setUp(self):
        self.n_users = 10
        self.n_items = 100
        self.random_state = 42

    def test_getDefaultConfigurationSpace_givenInvalidFeedback_valueErrorThrown(self):
        with self.assertRaises(ValueError) as cm:
            get_default_configuration_space(feedback="", n_users=self.n_users, n_items=self.n_items,
                                            random_state=self.random_state)
        self.assertEqual("Unknown feedback type: ", cm.exception.args[0])

    @patch('lkauto.utils.get_default_configuration_space.ItemItem', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.UserUser', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.FunkSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.BiasedSVD', model_mock)
    @patch('lkauto.utils.get_default_configuration_space.ImplicitMF', model_mock)
    def test_getDefaultConfigurationSpace_GivenImplicitAndValidInputs_CorrectConfigSpaceReturnedExpected(self):
        algorithm_list_string = 'ItemItem, FunkSVD, UserUser, ImplicitMF, BiasedSVD'

        result = get_default_configuration_space(feedback='implicit', n_users=self.n_users,
                                                 n_items=self.n_items,
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
        algorithm_list_string = 'ItemItem, UserUser, FunkSVD, BiasedSVD, ALSBiasedMF, Bias'

        result = get_default_configuration_space(feedback='explicit', n_users=self.n_users,
                                                 n_items=self.n_items,
                                                 random_state=self.random_state)

        self.assertIsInstance(result, ConfigurationSpace)
        self.assertTrue(algorithm_list_string in str(result.get('algo')))


if __name__ == '__main__':
    unittest.main()
