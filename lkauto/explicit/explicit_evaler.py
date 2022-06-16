import pandas as pd
from lenskit.metrics.predict import rmse
from lkauto.utils.get_model_from_cs import get_explicit_model_from_cs
import numpy as np


class ExplicitEvaler:

    def __init__(self, train, filer, random_seed=None, folds=None):
        self.train = train
        self.filer = filer
        self.random_seed = random_seed
        self.folds = folds
        self.run_id = 0
        self.top_50_runs = pd.DataFrame(columns=['run_id', 'model', 'error'])

    def __update_top_50_runs(self, config_space, root_mean_square_errors):
        model_performance = pd.DataFrame(data={'run_id': [self.run_id],
                                               'model': [config_space['regressor']],
                                               'error': [root_mean_square_errors.mean()]})
        if len(self.top_50_runs) < 50:
            self.top_50_runs = pd.concat([self.top_50_runs, model_performance])
        elif len(self.top_50_runs[self.top_50_runs['error'] > root_mean_square_errors.mean()]) > 0:
            max_val = self.top_50_runs['error'].max()
            self.top_50_runs = self.top_50_runs[self.top_50_runs['error'] < max_val]
            self.top_50_runs = pd.concat([self.top_50_runs, model_performance])

    def evaluate_explicit(self, config_space):
        output_path = 'smac_runs/'
        self.run_id += 1
        root_mean_square_errors = np.array([])
        validation_data = pd.DataFrame()

        # get model from configuration space
        model = get_explicit_model_from_cs(config_space)

        # holdout split using pandas and numpy rand
        validation_train = self.train.sample(frac=0.75, random_state=self.random_seed)  # random state is a seed value
        test = self.train.drop(validation_train.index)
        X_validation_test = test.copy()
        y_validation_test = test.copy()

        # preprocess validation split
        X_validation_test = X_validation_test.drop('rating', inplace=False, axis=1)
        y_validation_test = y_validation_test[['rating']].iloc[:, 0]

        # fit and predict model from configuration
        model.fit(validation_train)
        predictions = model.predict(X_validation_test)
        predictions.index = X_validation_test.index

        # calculate RMSE and append to numpy array
        root_mean_square_errors = np.append(root_mean_square_errors,
                                            rmse(predictions, y_validation_test, missing='ignore'))

        validation_data = pd.concat([validation_data, predictions], axis=0)

        # Save validation data for ensmables
        self.__update_top_50_runs(config_space=config_space, root_mean_square_errors=root_mean_square_errors)
        self.filer.save_validataion_data(config_space=config_space,
                                         predictions=validation_data,
                                         metric_scores=root_mean_square_errors,
                                         output_path=output_path,
                                         run_id=self.run_id)

        return root_mean_square_errors.mean()
