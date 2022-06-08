from lenskit.metrics.predict import rmse
from lkauto.utils.get_model_from_cs import get_explicit_model_from_cs
import lenskit.crossfold as xf
import numpy as np


class ExplicitEvaler:

    def __init__(self, train, random_seed=None, folds=None):
        self.train = train
        self.random_seed = random_seed
        self.folds = folds

    def evaluate_explicit(self, configuration_space):
        root_mean_square_errors = np.array([])
        model = get_explicit_model_from_cs(configuration_space)

        for i, tp in enumerate(xf.partition_rows(self.train, self.folds, rng_spec=self.random_seed)):
            train_vaidation_split = tp.train.copy()
            X_test_vaidation_split = tp.test.copy()
            X_test_vaidation_split.drop('rating', inplace=True, axis=1)
            y_test_validation_split = tp.test.copy()
            y_test_validation_split = y_test_validation_split[['rating']].iloc[:, 0]

            model.fit(train_vaidation_split)
            predictions = model.predict(X_test_vaidation_split)

            root_mean_square_errors = np.append(root_mean_square_errors,
                                                rmse(predictions, y_test_validation_split, missing='ignore'))

        return root_mean_square_errors.mean()
