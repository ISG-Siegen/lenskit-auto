import unittest
import pandas as pd

from unittest.mock import MagicMock, patch

from lenskit import Pipeline
from lenskit.funksvd import FunkSVDScorer
from lenskit.knn import ItemKNNScorer, UserKNNScorer
from lenskit.data import from_interactions_df
from lenskit.metrics import NDCG

from lkauto.utils.filer import Filer
from lkauto.algorithms.item_knn import ItemItem
from lkauto.algorithms.user_knn import UserUser
from lkauto.ensemble.ensemble_builder import get_pipelines_from_top_n_runs, build_ensemble, models_to_pipelines


class TestEnsembleBuilder(unittest.TestCase):

    def test_get_pipelines_from_top_n_runs_full(self):
        ii_pipeline = MagicMock(spec=ItemKNNScorer)
        ii_pipeline2 = MagicMock(spec=ItemKNNScorer)
        uu_pipeline = MagicMock(spec=UserKNNScorer)

        top_n_runs = pd.DataFrame({'pipeline': [ii_pipeline, ii_pipeline2, uu_pipeline]})

        pipelines = get_pipelines_from_top_n_runs(top_n_runs)

        self.assertEqual(len(pipelines), 3)
        self.assertEqual(ii_pipeline in pipelines, True)
        self.assertEqual(ii_pipeline2 in pipelines, True)
        self.assertEqual(uu_pipeline in pipelines, True)

    def test_get_pipelines_from_top_n_runs_empty(self):
        top_n_runs = pd.DataFrame({'pipeline': []})

        pipelines = get_pipelines_from_top_n_runs(top_n_runs)

        self.assertEqual(len(pipelines), 0)

    @patch('lkauto.utils.filer.Filer.get_dataframe_from_csv')
    @patch('lkauto.utils.filer.Filer.get_dict_from_json_file')
    def test_build_ensemble(self, mock_get_dataframe_from_csv, mock_get_dict_from_json):
        mock_get_dict_from_json.side_effect = [pd.DataFrame({'user_id': [1,2,3], 'item_id': [10,20,30], 'score': [1,2,3], 'rating': [1,2,3.5], 'timestamp': [MagicMock(), MagicMock(), MagicMock()]}),
            pd.DataFrame({'user_id': [1,2,3], 'item_id': [10,20,30], 'score': [2,3,4], 'rating': [1,2,3.5], 'timestamp': [MagicMock(), MagicMock(), MagicMock()]}),
            pd.DataFrame({'user_id': [1,2,3], 'item_id': [10,20,30], 'score': [3,4,5], 'rating': [1,2,3.5], 'timestamp': [MagicMock(), MagicMock(), MagicMock()]})
            ]

        mock_get_dataframe_from_csv.side_effect = [
            pd.DataFrame({'model': ['ItemKNN']}),
            pd.DataFrame({'model': ['UserKNN']}),
            pd.DataFrame({'model': ['FunkSVD']}),
        ]

        train = from_interactions_df(
            pd.DataFrame({'user_id': [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9],
                          'item_id': [10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                          'rating': [1,1,2.5,2.5,3.5,3.5,2,2,3.5,3.5,4.5,4.5,3,3,4.5,4.5,4.5,4.5]}),
        )
        top_n_runs = pd.DataFrame(
            {'run_id': [0,1,2], 'error': [0.5, 1, 2], 'pipeline': [MagicMock(spec=ItemKNNScorer), MagicMock(spec=UserKNNScorer), MagicMock(spec=FunkSVDScorer)]}
        )
        filer = Filer()

        es, incumbent = build_ensemble(train=train, top_n_runs=top_n_runs, filer=filer, ensemble_size=2, lenskit_metric=NDCG, maximize_metric=False)

        self.assertEqual(3, len(es.base_models))
        self.assertIn(top_n_runs.iloc[0]['pipeline'], es.base_models)
        self.assertIn(top_n_runs.iloc[1]['pipeline'], es.base_models)
        self.assertIn(top_n_runs.iloc[2]['pipeline'], es.base_models)

    @patch('lkauto.utils.filer.Filer.get_dataframe_from_csv')
    @patch('lkauto.utils.filer.Filer.get_dict_from_json_file')
    def test_build_ensemble_raise_Error(self, mock_get_dataframe_from_csv, mock_get_dict_from_json):
        mock_get_dict_from_json.side_effect = [pd.DataFrame({'user_id': [1, 2, 3], 'item_id': [10, 20, 30], 'score': [1, 2, 3], 'rating': [1, 2, 3.5], 'timestamp': [MagicMock(), MagicMock(), MagicMock()]}),
            pd.DataFrame({'user_id': [1, 4, 3], 'item_id': [10, 20, 30], 'score': [2, 3, 4], 'rating': [1, 2, 3.5], 'timestamp': [MagicMock(), MagicMock(), MagicMock()]}),
            pd.DataFrame({'user_id': [1, 2, 7], 'item_id': [10, 20, 30], 'score': [3, 4, 5], 'rating': [1, 2, 3.5], 'timestamp': [MagicMock(), MagicMock(), MagicMock()]})
            ]

        mock_get_dataframe_from_csv.side_effect = [
            pd.DataFrame({'model': ['ItemKNN']}),
            pd.DataFrame({'model': ['UserKNN']}),
            pd.DataFrame({'model': ['FunkSVD']}),
        ]

        train = from_interactions_df(
            pd.DataFrame({'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
                          'item_id': [10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                          'rating': [1, 1, 2.5, 2.5, 3.5, 3.5, 2, 2, 3.5, 3.5, 4.5, 4.5, 3, 3, 4.5, 4.5, 4.5, 4.5]}),
        )
        top_n_runs = pd.DataFrame(
            {'run_id': [0, 1, 2], 'error': [0.5, 1, 2],
             'pipeline': [MagicMock(spec=ItemKNNScorer), MagicMock(spec=UserKNNScorer), MagicMock(spec=FunkSVDScorer)]}
        )
        filer = Filer()

        self.assertRaises(ValueError, build_ensemble, train, top_n_runs, filer, ensemble_size=2, lenskit_metric=NDCG, maximize_metric=False)

    def test_models_to_pipeline(self):
        base_models = [ItemItem(max_nbrs=50), UserUser(max_nbrs=100)]

        models = models_to_pipelines(base_models)

        self.assertEqual(len(models), 2)
        self.assertIsInstance(models[0], Pipeline)
        self.assertIsInstance(models[1], Pipeline)

    def test_models_to_pipeline_empty(self):
        base_models = []

        models= models_to_pipelines(base_models)

        self.assertEqual(len(models), 0)




