import numpy as np
import pandas as pd
import time

from ConfigSpace import ConfigurationSpace, Configuration

from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.implicit.implicit_evaler import ImplicitEvaler
from lkauto.utils.filer import Filer

from typing import Tuple


def random_search(cs: ConfigurationSpace,
                  train: pd.DataFrame,
                  user_feedback: str,
                  optimization_metric,
                  filer: Filer,
                  time_limit_in_sec: int = 3600,
                  num_evaluations: int = None,
                  split_folds: int = 1,
                  split_strategie: str = 'user_based',
                  split_frac: float = 0.25,
                  ensemble_size: int = 50,
                  minimize_error_metric_val: bool = True,
                  num_recommendations: int = 10,
                  random_state=42) -> Tuple[Configuration, pd.DataFrame]:
    """
        returns the best configuration found by random search

         The random_search method will randomly search through the ConfigurationSpace to find the
         best performing configuration for the given train split. The ConfigurationSpace can consist of
         hyperparameters for a single algorithm or a combination of algorithms.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and hyperparameter ranges defined.
        num_evaluations: int
            number of samples to be randomly drawn from the ConfigurationSpace
        optimization_metric : function
            LensKit prediction accuracy metric to optimize for (either rmse or mae)
        minimize_error_metric_val : bool
            Bool that decides if the error metric should be minimized or maximized.
        user_feedback : str
            Defines if the dataset contains explicit or implicit feedback.
        random_state: int
        filer : Filer
            filer to manage LensKit-Auto output
        time_limit_in_sec
            time limit in seconds for the optimization process
        split_folds : int
            number of folds for the validation split cross validation
        split_strategie : str
            cross validation strategie (user_based or row_based)
        split_frac : float
            fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
            will be ignored.
        ensemble_size : int
            number of models to be used in the ensemble of rating prediction tasks. This value will be ignored
            for recommender tasks.
        num_recommendations : int
            number of recommendations to be made for each user. This value will be ignored for rating prediction.

        Returns
        -------
        best_configuration : Configuration
            the best suited (algorithm and) hyperparameter configuration for the train dataset.
        top_n_runs : pd.DataFrame
            pandas dataframe containing the top n runs of the random search.
   """
    # initialize variables to keep track of the best configuration
    best_configuration = None
    if minimize_error_metric_val:
        best_error_score = np.inf
    else:
        best_error_score = 0

    # initialize evaler based on user feedback
    if user_feedback == "explicit":
        evaler = ExplicitEvaler(train=train,
                                optimization_metric=optimization_metric,
                                filer=filer,
                                random_state=random_state,
                                split_folds=split_folds,
                                split_strategie=split_strategie,
                                split_frac=split_frac,
                                ensemble_size=ensemble_size,
                                minimize_error_metric_val=minimize_error_metric_val)
    elif user_feedback == 'implicit':
        evaler = ImplicitEvaler(train=train,
                                optimization_metric=optimization_metric,
                                filer=filer,
                                random_state=random_state,
                                split_folds=split_folds,
                                split_strategie=split_strategie,
                                split_frac=split_frac,
                                num_recommendations=num_recommendations,
                                minimize_error_metric_val=minimize_error_metric_val)
    else:
        raise ValueError('feedback must be either explicit or implicit')

    # random sample n_samples configurations from the ConfigurationSpace. Store them in a set to avoid duplicates.

    # run for a specified number of iterations or until time limit is reached
    if num_evaluations is None:
        # track time to support random_search on a time base
        start_time = time.time()

        # track configurations that have already been tested
        configuration_list = []

        # loop through random spampled configurations
        while time.time() - start_time > time_limit_in_sec:
            # random sample configuration from configuration space
            config = cs.sample_configuration()
            # check if configuration has already been tested
            if config not in configuration_list:
                # calculate error for the configuration
                error = evaler.evaluate(config)

                # keep track of best performing configuration
                if minimize_error_metric_val:
                    if error < best_error_score:
                        best_error_score = error
                        best_configuration = config
                else:
                    if error > best_error_score:
                        best_error_score = error
                        best_configuration = config
    else:
        configuration_set = set()
        while len(configuration_set) < num_evaluations:
            configuration_set.add(cs.sample_configuration())

        # track time to support random_search on a time base
        start_time = time.time()

        # loop through random spampled configurations
        for config in configuration_set:
            # calculate error for the configuration
            error = evaler.evaluate(config)

            # keep track of best performing configuration
            if minimize_error_metric_val:
                if error < best_error_score:
                    best_error_score = error
                    best_configuration = config
            else:
                if error > best_error_score:
                    best_error_score = error
                    best_configuration = config

            # keep track of time
            if time.time() - start_time > time_limit_in_sec:
                break

    if user_feedback == "explicit":
        filer.save_dataframe_as_csv(evaler.top_n_runs, '', 'top_n_runs')
        return best_configuration, evaler.top_n_runs
    elif user_feedback == 'implicit':
        return best_configuration
    else:
        raise ValueError('feedback must be either explicit or implicit')
