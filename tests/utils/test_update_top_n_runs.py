import unittest

import pandas as pd
import numpy as np

from unittest.mock import MagicMock

from lkauto.utils.update_top_n_runs import update_top_n_runs

class TestUpdateTopNRuns(unittest.TestCase):

    def setUp(self):
        ii_pipeline = MagicMock()
        ii_pipeline2 = MagicMock()
        uu_pipeline = MagicMock()
        b_pipeline = MagicMock()
        self.top_n_runs = pd.DataFrame({"run_id": [1,2,3,4],
                                        "model": ["ItemItem", "ItemItem", "UserUser", "Bias"],
                                        "error": [0.51, 0.52, 0.53, 0.54],
                                        "pipeline": [ii_pipeline, ii_pipeline2, uu_pipeline, b_pipeline]})

    def test_updateTopNRuns_replace_one_run(self):
        configSpace = MagicMock()
        configSpace.__getitem__.return_value = "FunkSVD"
        pipeline = MagicMock()
        new_top_n_runs = update_top_n_runs(4, self.top_n_runs, 5, configSpace, pipeline, np.array([0.54, 0.55, 0.56]))

        self.assertTrue(new_top_n_runs.shape[0] == 4)
        self.assertTrue(new_top_n_runs.loc[new_top_n_runs["run_id"] == 5, "model"].iloc[0] == "FunkSVD")
        # run with id = 2 should be removed, because ItemItem has 2 runs, und id = 2 has a bigger error
        self.assertFalse(2 in new_top_n_runs["run_id"].values)

    def test_updateTopNRuns_add_one_run(self):
        configSpace = MagicMock()
        configSpace.__getitem__.return_value = "FunkSVD"
        pipeline = MagicMock()
        new_top_n_runs = update_top_n_runs(5, self.top_n_runs, 5, configSpace, pipeline, np.array([0.54, 0.55, 0.56]))

        self.assertTrue(new_top_n_runs.shape[0] == 5)
        self.assertTrue(new_top_n_runs.loc[new_top_n_runs["run_id"] == 5, "model"].iloc[0] == "FunkSVD")

    def test_updateTopNRuns_dont_add_run_with_most_runs_already(self):
        configSpace = MagicMock()
        configSpace.__getitem__.return_value = "ItemItem"
        pipeline = MagicMock()
        new_top_n_runs = update_top_n_runs(4, self.top_n_runs, 5, configSpace, pipeline, np.array([0.54, 0.55, 0.56]))

        self.assertTrue(new_top_n_runs.shape[0] == 4)
        self.assertFalse(5 in new_top_n_runs["run_id"].values)

    def test_updateTopNRuns_replace_run_with_most_runs_already(self):
        configSpace = MagicMock()
        configSpace.__getitem__.return_value = "ItemItem"
        pipeline = MagicMock()
        new_top_n_runs = update_top_n_runs(4, self.top_n_runs, 5, configSpace, pipeline, np.array([0.44, 0.45, 0.46]))

        self.assertTrue(new_top_n_runs.shape[0] == 4)
        # run 2 gets replaced by run 5
        self.assertTrue(5 in new_top_n_runs["run_id"].values)
        self.assertTrue(new_top_n_runs.loc[new_top_n_runs["run_id"] == 5, "model"].iloc[0] == "ItemItem")
        self.assertFalse(2 in new_top_n_runs["run_id"].values)
