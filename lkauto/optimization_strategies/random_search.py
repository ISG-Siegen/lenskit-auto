import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from lkauto.utils.get_model_from_cs import get_model_from_cs
from lenskit.metrics.predict import rmse
import numpy as np


def random_search(cs: ConfigurationSpace, train: pd.DataFrame, n_samples: int,
                  minimize_error_metric_val: bool = True, user_feedback: str = "explicit",
                  random_state=42) -> Configuration:
    """ returns the best configuration found by random search

         The random_search method will randomly search through the ConfigurationSpace to find the
         best performing configuration for the given train split. The ConfigurationSpace can consist of
         hyperparameters for a single algorithm or a combination of algorithms.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and hyperparameter ranges defined.
        n_samples: int
            number of samples to be randomly drawn from the ConfigurationSpace
        minimize_error_metric_val : bool
            Bool that decides if the error metric should be minimized or maximized.
        user_feedback : str
            Defines if the dataset contains explicit or implicit feedback.
        random_state: int

        Returns
        -------
        best_configuration : Configuration
            the best suited (algorithm and) hyperparameter configuration for the train dataset.
        val_error_score : float
            best validation error found on the validation split
   """
    best_configuraiton = None
    best_error_score = np.inf

    # random sample n_samples configurations from the ConfigurationSpace. Store them in a set to avoid duplicates.
    configuration_set = set()
    while len(configuration_set) < n_samples:
        configuration_set.add(cs.sample_configuration())

    for config in configuration_set:

        model = get_model_from_cs(config, feedback=user_feedback)

        model.fit(train)

        # holdout split using pandas and numpy random seed
        validation_train = train.sample(frac=0.75, random_state=random_state)  # random state is a seed value
        test = train.drop(validation_train.index)
        X_validation_test = test.copy()
        y_validation_test = test.copy()

        # process validation split
        X_validation_test = X_validation_test.drop('rating', inplace=False, axis=1)
        y_validation_test = y_validation_test[['rating']].iloc[:, 0]

        # fit and predict model from configuration
        model.fit(validation_train)
        predictions = model.predict(X_validation_test)
        predictions.index = X_validation_test.index

        # calculate error_metric
        error = rmse(predictions, y_validation_test, missing='ignore')

        if minimize_error_metric_val:
            if error < best_error_score:
                best_error_score = error
                best_configuraiton = config
        else:
            if error > best_error_score:
                best_error_score = error
                best_configuraiton = config

    return best_configuraiton
