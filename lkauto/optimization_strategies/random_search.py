import numpy as np
import pandas as pd
import time

from ConfigSpace import ConfigurationSpace, Configuration

from lenskit.data import Dataset, ItemListCollection

from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.implicit.implicit_evaler import ImplicitEvaler
from lkauto.utils.get_default_configurations import get_default_configurations
from lkauto.utils.filer import Filer
from lkauto.utils.get_default_configuration_space import get_default_configuration_space

from typing import Tuple
import logging


def random_search(cs: ConfigurationSpace,
                  train: Dataset,
                  user_feedback: str,
                  optimization_metric,
                  filer: Filer,
                  validation: ItemListCollection = None,
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
        validation : pd.DataFrame
            Pandas Dataframe validation split.
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
    logger = logging.getLogger('lenskit-auto')
    logger.info('--Start Random Search--')

    # initialize variables to keep track of the best configuration
    best_configuration = None
    if minimize_error_metric_val:
        best_error_score = 0
    else:
        best_error_score = np.inf

    # initialize evaler based on user feedback
    if user_feedback == "explicit":
        evaler = ExplicitEvaler(train=train,
                                optimization_metric=optimization_metric,
                                filer=filer,
                                validation=validation,
                                random_state=random_state,
                                split_folds=split_folds,
                                split_strategy=split_strategie,
                                split_frac=split_frac,
                                ensemble_size=ensemble_size,
                                minimize_error_metric_val=minimize_error_metric_val)
    elif user_feedback == 'implicit':
        evaler = ImplicitEvaler(train=train,
                                optimization_metric=optimization_metric,
                                filer=filer,
                                validation=validation,
                                random_state=random_state,
                                split_folds=split_folds,
                                split_strategy=split_strategie,
                                split_frac=split_frac,
                                num_recommendations=num_recommendations,
                                minimize_error_metric_val=minimize_error_metric_val)
    else:
        raise ValueError('feedback must be either explicit or implicit')

    # get pre-defined ConfiguraitonSpace if none is provided
    if cs is None:
        logger.debug('initializing default ConfigurationSpace')
        cs = get_default_configuration_space(data=train,
                                             val_fold_indices=evaler.train_test_splits,
                                             validation=validation,
                                             feedback='explicit',
                                             random_state=random_state)

    # get LensKit default configurations
    initial_configuraition = get_default_configurations(cs)

    # run for a specified number of iterations or until time limit is reached
    if num_evaluations == 0:
        # track time to support random_search on a time base
        start_time = time.time()

        # track configurations that have already been tested
        configuration_list = []

        # loop through random spampled configurations
        while time.time() - start_time < time_limit_in_sec:
            # initialize config
            config = None
            # check if all initial configurations have been tested
            if all(x in configuration_list for x in initial_configuraition):
                # random sample configuration from configuration space
                config = cs.sample_configuration()
            else:
                # pop configuration from initial configurations
                config = initial_configuraition.pop(0)

            # check if configuration has already been tested
            if config not in configuration_list:
                # calculate error for the configuration
                error = evaler.evaluate(config)

                # keep track of best performing configuration
                if minimize_error_metric_val:
                    if error > best_error_score:
                        best_error_score = error
                        best_configuration = config
                else:
                    if error < best_error_score:
                        best_error_score = error
                        best_configuration = config
                # add configuration to list of tested configurations
                configuration_list.append(config)
    else:
        configuration_set = set()

        # add initial configurations to set
        for config in initial_configuraition:
            if len(configuration_set) < num_evaluations:
                configuration_set.add(config)

        # add configurations to set until the set contains num_evaluations configurations
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
                if error > best_error_score:
                    best_error_score = error
                    best_configuration = config
            else:
                if error < best_error_score:
                    best_error_score = error
                    best_configuration = config

            # keep track of time
            if time.time() - start_time > time_limit_in_sec:
                break

    logger.info('--End Random Search--')

    if user_feedback == "explicit":
        filer.save_dataframe_as_csv(evaler.top_n_runs, '', 'top_n_runs')
        return best_configuration, evaler.top_n_runs
    elif user_feedback == 'implicit':
        return best_configuration
    else:
        raise ValueError('feedback must be either explicit or implicit')
