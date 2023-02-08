import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration

from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.implicit.implicit_evaler import ImplicitEvaler
from lkauto.utils.filer import Filer


def random_search(cs: ConfigurationSpace,
                  train: pd.DataFrame,
                  n_samples: int,
                  filer: Filer,
                  optimization_metric,
                  user_feedback: str = None,
                  minimize_error_metric_val: bool = True,
                  random_state=42) -> Configuration:
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
        n_samples: int
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

        Returns
        -------
        best_configuration : Configuration
            the best suited (algorithm and) hyperparameter configuration for the train dataset.
        val_error_score : float
            best validation error found on the validation split
   """
    best_configuration = None
    best_error_score = np.inf

    if user_feedback == "explicit":
        evaler = ExplicitEvaler(train=train,
                                filer=filer,
                                optimization_metric=optimization_metric,
                                random_state=random_state)
    elif user_feedback == 'implicit':
        evaler = ImplicitEvaler(train=train,
                                filer=filer,
                                optimization_metric=optimization_metric,
                                random_state=random_state)
    else:
        raise ValueError('feedback must be either explicit or implicit')

    # random sample n_samples configurations from the ConfigurationSpace. Store them in a set to avoid duplicates.
    configuration_set = set()
    while len(configuration_set) < n_samples:
        configuration_set.add(cs.sample_configuration())

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

    return best_configuration
