import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from unittest.mock import MagicMock

from lenskit.batch import predict
from lenskit.funksvd import FunkSVDScorer
from lenskit.metrics import RMSE
from lenskit.data import ItemList, load_movielens, ItemListCollection
from lenskit.knn.item import ItemKNNScorer
from lenskit.knn.user import UserKNNScorer
from lenskit.pipeline import predict_pipeline
from lenskit.splitting import sample_records

from lkauto.ensemble.greedy_ensemble_selection import EnsembleSelection

class TestGreedyEnsembleSelection(unittest.TestCase):

    def setUp(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        ml_folder = project_root / "data" / "ml-latest-small"
        ml_dataset = load_movielens(ml_folder)
        ttsplit = sample_records(ml_dataset, size=50)
        self.train = ttsplit.train
        self.test = ttsplit.test

    def test_minimized_metric_with_minimize(self):
        y_pred = ItemList.from_df(pd.DataFrame({'item_id': [0,1,2,3,4], 'scores': [1,2,3,4,5]}))
        y_true = ItemList.from_df(pd.DataFrame({'item_id': [0,1,2,3,4], 'rating': [1.5, 2, 3, 5, 4.5]}))
        metric = RMSE()
        true_result = metric.measure_list(y_pred, y_true)

        ensemble_selection = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        test_result = ensemble_selection.minimized_metric(y_true, y_pred)

        self.assertEqual(true_result, test_result)

    def test_minimized_metric_with_maximize(self):
        y_pred = ItemList.from_df(pd.DataFrame({'item_id': [0, 1, 2, 3, 4], 'scores': [1, 2, 3, 4, 5]}))
        y_true = ItemList.from_df(pd.DataFrame({'item_id': [0, 1, 2, 3, 4], 'rating': [1.5, 2, 3, 5, 4.5]}))
        metric = RMSE()
        true_result = -metric.measure_list(y_pred, y_true)

        ensemble_selection = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=True)
        test_result = ensemble_selection.minimized_metric(y_true, y_pred)

        self.assertEqual(true_result, test_result)

    def test_fit_no_base_models(self):
        es = EnsembleSelection(ensemble_size=2, lenskit_metric=RMSE, maximize_metric=True)
        es.base_models = None

        self.assertRaises(ValueError, es.fit, MagicMock())

    def test_fit(self):
        es = EnsembleSelection(ensemble_size=2, lenskit_metric=RMSE, maximize_metric=True)
        itemknn = predict_pipeline(ItemKNNScorer())
        userknn = predict_pipeline(UserKNNScorer())
        es.base_models = [itemknn, userknn]

        # try predicting without fitting. The predict function cannot be
        # used with a pipeline that is not fitted, so this should throw
        # an exception
        with self.assertRaises(Exception):
            preds_i_before = predict(itemknn, self.test)
        with self.assertRaises(Exception):
            preds_u_before = predict(userknn, self.test)

        es.fit(self.train)

        # predict after fitting
        preds_i_after = predict(itemknn, self.test)
        preds_u_after = predict(userknn, self.test)

        self.assertIsInstance(preds_i_after, ItemListCollection)
        self.assertEqual(len(preds_i_after), len(self.test))
        self.assertIsInstance(preds_u_after, ItemListCollection)
        self.assertEqual(len(preds_u_after), len(self.test))

    def test_predict(self):
        es = EnsembleSelection(ensemble_size=2, lenskit_metric=RMSE, maximize_metric=True)
        itemknn = predict_pipeline(ItemKNNScorer())
        userknn = predict_pipeline(UserKNNScorer())
        es.base_models = [itemknn, userknn]

        es.fit(self.train)

        itemknn_preds = predict(itemknn, self.test)
        userknn_preds = predict(userknn, self.test)
        predictions_set = [itemknn_preds, userknn_preds]
        labels = self.test.to_df()["rating"]
        print(labels)
        predictions_set = []
        itemknn_preds_df = itemknn_preds.to_df()
        userknn_preds_df = userknn_preds.to_df()
        predictions_set.append(np.array(itemknn_preds_df[list(itemknn_preds_df)[1]]))
        predictions_set.append(np.array(userknn_preds_df[list(userknn_preds_df)[1]]))
        es.ensemble_fit(predictions_set, labels)

        preds = es.predict(self.test)

        self.assertIsInstance(preds, ItemListCollection)
        self.assertEqual(len(preds), len(self.test))

    def test_fast(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)

        preds = [np.array([2.5,3,3,4,2]),
                 np.array([2.5,2.5,3.5,4,1.5]),
                 np.array([1.5, 3, 3.5, 5, 3])]
        labels = np.array([2,3,4,5,1])

        es._fast(predictions=preds, labels=labels)

        self.assertEqual(es.indices_, [1,1,2])
        for actual, expected in zip(es.trajectory_, [0.63, 0.63, 0.6]):
            self.assertAlmostEqual(actual, expected, delta=0.1)
        self.assertAlmostEqual(es.train_loss_, 0.6, delta=0.1)

    def test_fast_emtpy_preds(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)

        preds = []
        labels = np.array([2,3,4,5,1])

        self.assertRaises(ValueError, es._fast, predictions=preds, labels=labels)

    def test_fast_emtpy_labels(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)

        preds = [np.array([2.5, 3, 3, 4, 2]),
                 np.array([2.5, 2.5, 3.5, 4, 1.5]),
                 np.array([1.5, 3, 3.5, 5, 3])]
        labels = np.array([])

        self.assertRaises(ValueError, es._fast, predictions=preds, labels=labels)

    def test_fast_preds_labels_mismatch(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)

        preds = [np.array([2.5, 3, 3, 4, 2]),
                 np.array([2.5, 2.5, 3.5, 4, 1.5]),
                 np.array([1.5, 3, 3.5, 5, 3])]
        labels = np.array([2,3])

        self.assertRaises(ValueError, es._fast, predictions=preds, labels=labels)

    def test_fast_single_pred(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)

        preds = [np.array([2.5, 2.5, 3.5, 4, 1.5])]
        labels = np.array([2,3,4,5,1])

        es._fast(preds, labels)

        self.assertEqual(es.indices_, [0])
        self.assertAlmostEqual(es.trajectory_[0], 0.63, delta=0.1)
        self.assertAlmostEqual(es.train_loss_, 0.63, delta=0.1)

    def test_apply_use_best(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        es.indices_ = [1,1,2,3]
        es.trajectory_ = [0.63, 0.63, 0.6, 0.65]

        es._apply_use_best()

        self.assertEqual(es.indices_, [1,1,2])
        self.assertEqual(es.trajectory_, [0.63, 0.63, 0.6])
        self.assertEqual(es.ensemble_size, 3)
        self.assertEqual(es.train_loss_, 0.6)

    def test_apply_use_best_first_is_best(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        es.indices_ = [1, 2, 3, 4]
        es.trajectory_ = [0.6, 0.63, 0.64, 0.65]

        es._apply_use_best()

        self.assertEqual(es.indices_, [1])
        self.assertEqual(es.trajectory_, [0.6])
        self.assertEqual(es.ensemble_size, 1)
        self.assertEqual(es.train_loss_, 0.6)

    def test_calculate_weights(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        es.indices_ = [1,1,2,3]
        es.num_input_models_ = 10

        es._calculate_weights()

        expected_weights = np.array([0, 2/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.array_equal(es.weights_, expected_weights))

    def test_calculate_weights_sum_under_one(self):
        es = EnsembleSelection(ensemble_size=5, lenskit_metric=RMSE, maximize_metric=False)
        es.indices_ = [1, 1, 2, 3]
        es.num_input_models_ = 10

        es._calculate_weights()

        expected_weights = np.array([0, 2/5, 1/ 5, 1/3, 0, 0, 0, 0, 0, 0])
        adjusted_expected_weigths = np.array([0, 10/20, 5/20, 5/20, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.array_equal(es.weights_, adjusted_expected_weigths))

    def test_ensemble_fit_too_small_ensemble(self):
        es = EnsembleSelection(ensemble_size=0, lenskit_metric=RMSE, maximize_metric=False)
        preds = [np.array([2.5, 2.5, 3.5, 4, 1.5])]
        labels = np.array([2, 3, 4, 5, 1])

        self.assertRaises(ValueError, es.ensemble_fit, preds, labels)

    def test_ensemble_fit(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        preds = [np.array([2.5, 3, 3, 4, 2]),
                 np.array([2.5, 2.5, 3.5, 4, 1.5]),
                 np.array([1.5, 3, 3.5, 5, 3])]
        labels = np.array([2, 3, 4, 5, 1])

        es.ensemble_fit(preds, labels)

        self.assertEqual(es.indices_, [1, 1, 2])
        for actual, expected in zip(es.trajectory_, [0.63, 0.63, 0.6]):
            self.assertAlmostEqual(actual, expected, delta=0.1)
        self.assertAlmostEqual(es.train_loss_, 0.6, delta=0.1)
        self.assertEqual(es.ensemble_size, 3)
        self.assertAlmostEqual(es.train_loss_, 0.6, delta=0.1)
        expected_weights = np.array([0, 2/3, 1/3])
        self.assertTrue(np.array_equal(es.weights_, expected_weights))

    def test_ensemble_predict(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        es.weights_ = np.array([0, 2/3, 1/3])
        preds = [np.array([[1, 1, 2.5, 2, MagicMock()], [2, 2, 3, 3.5, MagicMock()], [3, 3, 3, 3.5, MagicMock()], [4, 4, 2, 2.5, MagicMock()]], dtype=object),
                 np.array([[1, 1, 2.5, 2, MagicMock()], [2, 2, 2.5, 3.5, MagicMock()], [3, 3, 3.5, 3.5, MagicMock()], [4, 4, 1.5, 2.5, MagicMock()]], dtype=object),
                 np.array([[1, 1, 1.5, 2, MagicMock()], [2, 2, 3, 3.5, MagicMock()], [3, 3, 3.5, 3.5, MagicMock()], [4, 4, 5, 2.5, MagicMock()]], dtype=object)]

        ensemble_preds = es.ensemble_predict(preds)

        expected_predictions = np.array([2.166, 2.66, 3.5, 2.66])

        for actual, expected in zip(ensemble_preds, expected_predictions):
            self.assertAlmostEqual(actual, expected, delta=0.05)

    def test_ensemble_predict_len_predictions_equals_non_zero_weights(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        es.weights_ = np.array([0, 2 / 3, 1 / 3, 0, 0], )
        preds = [np.array([[1, 1, 2.5, 2, MagicMock()], [2, 2, 3, 3.5, MagicMock()], [3, 3, 3, 3.5, MagicMock()], [4, 4, 2, 2.5, MagicMock()]], dtype=object),
                 np.array([[1, 1, 2.5, 2, MagicMock()], [2, 2, 2.5, 3.5, MagicMock()], [3, 3, 3.5, 3.5, MagicMock()], [4, 4, 1.5, 2.5, MagicMock()]], dtype=object)]

        ensemble_preds = es.ensemble_predict(preds)

        expected_predictions = np.array([2.5, 2.83, 3.16, 1.83])

        for actual, expected in zip(ensemble_preds, expected_predictions):
            print(f"Actual: {actual}, Expected: {expected}")
            self.assertAlmostEqual(actual, expected, delta=0.05)

    def test_ensemble_predict_predictions_weights_mismatch(self):
        es = EnsembleSelection(ensemble_size=3, lenskit_metric=RMSE, maximize_metric=False)
        es.weights_ = np.array([0, 2 / 3, 1 / 3, 0, 0, 0, 0])
        preds = [np.array([[1, 1, 2.5, 2, MagicMock()], [2, 2, 3, 3.5, MagicMock()], [3, 3, 3, 3.5, MagicMock()], [4, 4, 2, 2.5, MagicMock()]], dtype=object),
                 np.array([[1, 1, 2.5, 2, MagicMock()], [2, 2, 2.5, 3.5, MagicMock()], [3, 3, 3.5, 3.5, MagicMock()], [4, 4, 1.5, 2.5, MagicMock()]], dtype=object),
                 np.array([[1, 1, 1.5, 2, MagicMock()], [2, 2, 3, 3.5, MagicMock()], [3, 3, 3.5, 3.5, MagicMock()], [4, 4, 5, 2.5, MagicMock()]], dtype=object)]

        self.assertRaises(ValueError, es.ensemble_predict, preds)
