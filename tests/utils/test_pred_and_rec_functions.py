import unittest

from unittest.mock import MagicMock, patch

from lkauto.utils.pred_and_rec_functions import predict, recommend
from lkauto.ensemble.ensemble_builder import EnsembleSelection
from lenskit.pipeline import Pipeline
from lenskit.data import ItemListCollection


class TestPredAndRecFunctions(unittest.TestCase):

    def setUp(self):
        self.test_split = MagicMock(spec=ItemListCollection)

    @patch('lkauto.utils.pred_and_rec_functions.lk_predict')
    def test_single_predict(self, mock_lk_predict):
        model = MagicMock(spec=Pipeline)
        expected_result = "Single Prediction"
        mock_lk_predict.return_value = expected_result

        result = predict(model, self.test_split)

        mock_lk_predict.assert_called_once_with(model, self.test_split)
        self.assertEqual(result, expected_result)

    def test_ensemble_predict(self):
        ensemble = MagicMock(spec=EnsembleSelection)
        expected_result = "Ensemble Prediction"
        ensemble.predict.return_value = expected_result

        result = predict(ensemble, self.test_split)

        # checks that the predict function of EnsembleSelection gets called once with the correct parameters
        ensemble.predict.assert_called_once_with(x_data=self.test_split)
        self.assertEqual(result, expected_result)

    def test_invalid_model_predict(self):
        model = "just a strig, not a model"

        self.assertRaises(TypeError, predict, model, self.test_split)

    @patch('lkauto.utils.pred_and_rec_functions.lk_recommend')
    def test_recommend(self, mock_lk_recommend):
        model = MagicMock(spec=Pipeline)
        expected_result = "Recommendation"
        mock_lk_recommend.return_value = expected_result

        result = recommend(model, self.test_split)

        mock_lk_recommend.assert_called_once_with(model, self.test_split)
        self.assertEqual(result, expected_result)

    def test_invalid_model_recommend(self):
        model = "just a strig, not a model"

        self.assertRaises(TypeError, recommend, model, self.test_split)
