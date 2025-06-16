import unittest
import uuid
from unittest.mock import MagicMock, patch

from lkauto.utils.get_model_from_cs import get_model_from_cs


class TestGetModelFromCS(unittest.TestCase):

    def setUp(self):
        self.cs = MagicMock()
        self.fallback_model = MagicMock()
        self.random_state = 42

    def test_getModelFromCS_givenInvalidFeedback_valueErrorThrown(self):
        with self.assertRaises(ValueError) as cm:
            get_model_from_cs(cs=self.cs, random_state=self.random_state,
                              feedback="")
        self.assertEqual("Unknown feedback type: ", cm.exception.args[0])

    def test_getModelFromCS_givenInvalidAlgorithmInCS_valueErrorThrown(self):
        self.cs.get.return_value = 'alg'
        self.cs.items.return_value = [('algo', 'alg'), ('alg:attr1', 'val1'), ('alg:attr2', 'val2')]

        with self.assertRaises(ValueError) as cm:
            get_model_from_cs(cs=self.cs, random_state=self.random_state,
                              feedback="implicit")
        self.assertEqual("Unknown algorithm: alg", cm.exception.args[0])

    def test_getModelFromCS_givenImplicitAndValidInputs_correctModelReturnedExpected(self):
        algorithm_params_list = [
            ('ItemItem', [('algo', 'ItemItem'), ('ItemItem:attr1', 'val1')]),
            ('UserUser', [('algo', 'UserUser'), ('UserUser:attr1', 'val1')]),
            ('FunkSVD', [('algo', 'FunkSVD'), ('FunkSVD:attr1', 'val1')]),
            ('BiasedSVD', [('algo', 'BiasedSVD'), ('BiasedSVD:attr1', 'val1')]),
            ('ALSBiasedMF', [('algo', 'ALSBiasedMF'), ('ALSBiasedMF:ureg', 1), ('ALSBiasedMF:ireg', 2)]),
            ('Bias', [('algo', 'Bias'), ('Bias:user_damping', 'val1'), ('Bias:item_damping', 'val2')]),
            ('ImplicitMF', [('algo', 'ImplicitMF'), ('ImplicitMF:ureg', 'val1'), ('ImplicitMF:ireg', 'val2')])]

        for algorithm_params in algorithm_params_list:
            with self.subTest(algorithm=algorithm_params[0]):
                self.cs.get.return_value = algorithm_params[0]
                self.cs.items.return_value = algorithm_params[1]
                algorithm_mock = MagicMock()
                algorithm_mock_test_id = uuid.uuid4()
                algorithm_mock.test_id = algorithm_mock_test_id

                with patch('lkauto.utils.get_model_from_cs.{}'.format(
                    algorithm_params[0] if algorithm_params[0] != 'ALSBiasedMF' else 'BiasedMF'),
                    return_value=algorithm_mock):
                    result = get_model_from_cs(cs=self.cs,
                                               random_state=self.random_state,
                                               feedback="implicit")

                self.assertEqual(algorithm_mock_test_id, result.predictor.test_id)

    def test_getModelFromCS_givenExplicitAndValidInputs_correctModelReturnedExpected(self):
        algorithm_params_list = [
            ('ItemItem', [('algo', 'ItemItem'), ('ItemItem:attr1', 'val1')]),
            ('UserUser', [('algo', 'UserUser'), ('UserUser:attr1', 'val1')]),
            ('FunkSVD', [('algo', 'FunkSVD'), ('FunkSVD:attr1', 'val1')]),
            ('BiasedSVD', [('algo', 'BiasedSVD'), ('BiasedSVD:attr1', 'val1')]),
            ('ALSBiasedMF', [('algo', 'ALSBiasedMF'), ('ALSBiasedMF:ureg', 1), ('ALSBiasedMF:ireg', 2)]),
            ('Bias', [('algo', 'Bias'), ('Bias:user_damping', 'val1'), ('Bias:item_damping', 'val2')]),
            ('ImplicitMF', [('algo', 'ImplicitMF'), ('ImplicitMF:ureg', 'val1'), ('ImplicitMF:ireg', 'val2')])]

        for algorithm_params in algorithm_params_list:
            with self.subTest(algorithm=algorithm_params[0]):
                self.cs.get.return_value = algorithm_params[0]
                self.cs.items.return_value = algorithm_params[1]
                algorithm_mock = MagicMock()
                algorithm_mock_test_id = uuid.uuid4()
                algorithm_mock.test_id = algorithm_mock_test_id

                with patch('lkauto.utils.get_model_from_cs.{}'.format(
                    algorithm_params[0] if algorithm_params[0] != 'ALSBiasedMF' else 'BiasedMF'),
                    return_value=algorithm_mock):
                    result = get_model_from_cs(cs=self.cs, fallback_model=self.fallback_model,
                                               random_state=self.random_state,
                                               feedback="explicit")

                self.assertEqual(algorithm_mock_test_id, result.algorithms[0].test_id)


if __name__ == '__main__':
    unittest.main()
