from lkauto.utils.get_model_from_cs import get_implicit_recommender_from_cs
import lenskit.crossfold as xf
from lenskit import topn, batch
import numpy as np
import pandas as pd


class ImplicitEvaler:
    def __init__(self, train, random_state, folds):
        self.train = train
        self.random_seed = random_state
        self.folds = folds

    def evaluate_implicit(self, configuration_space):
        precisions = np.array([])
        model = get_implicit_recommender_from_cs(configuration_space)

        for i, tp in enumerate(xf.partition_rows(self.train, self.folds, rng_spec=self.random_seed)):
            train_vaidation_split = tp.train.copy()
            X_test_vaidation_split = tp.test.copy()
            X_test_vaidation_split.drop('rating', inplace=True, axis=1)
            y_test_validation_split = tp.test.copy()
            y_test_validation_split = y_test_validation_split[['rating']].iloc[:, 0]

            model.fit(train_vaidation_split)
            recs = batch.recommend(model, X_test_vaidation_split['user'], 10, n_jobs=1)

            rla = topn.RecListAnalysis()
            rla.add_metric(topn.precision)

            scores = rla.compute(recs, y_test_validation_split)

            overall_precision = pd.Series.mean(scores['precision'])
            np.append(precisions, overall_precision)

        return 1-precisions.mean()
