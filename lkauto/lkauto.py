import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace

from lkauto.utils.get_model_from_cs import get_model_from_cs
from lkauto.utils.get_default_configuration_space import get_default_configuration_space
from lkauto.optimization_strategies.bayesian_optimization import bayesian_optimization
from lkauto.optimization_strategies.random_search import random_search
from lkauto.utils.filer import Filer

from lenskit.metrics.predict import rmse
from lenskit.metrics.topn import ndcg
from lenskit.algorithms import Predictor
from lenskit import Recommender

from typing import Tuple


def find_best_explicit_configuration(train: pd.DataFrame,
                                     cs: ConfigurationSpace = None,
                                     optimization_metric=rmse,
                                     optimization_strategie: str = 'bayesian',
                                     time_limit_in_sec: int = 2700,
                                     n_trials: int = 100,
                                     random_state=None,
                                     folds: int = 5,
                                     filer: Filer = None) -> Tuple[Predictor, dict]:
    """
        returns the best Predictor found in the defined search time

         the find_best_explicit_configuration method will search the ConfigurationSpace
         for the best Predictor model configuration.
         Depending on the ConfigurationSpace parameter provided by the developer,
         performs three different use-cases.
         1. combined algorithm selection and hyperparameter configuration
         2. combined algorthm selection and hyperparameter configuration for a specific subset
            of algorithms and/or different parameter ranges
         3. hyperparameter selection for a specific algorithm.
         The hyperparameter and/or model selection process will be stopped after the time_limit_in_sec or (if set) after
         n_trials. The first one to be reached will stop the optimization.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and parameter ranges defined.
        optimization_metric : function
            LensKit prediction accuracy metric to optimize for (either rmse or mae)
        optimization_strategie: str
            optimization strategie to use. Either bayesian or random_search
        time_limit_in_sec : int
            optimization search time limit in sec.
        n_trials : int
                number of samples to be used for optimization_strategy. Value can not be smaller than 6
                if no initial configuration is provided.
        random_state
            The random number generator or seed (see :py:func:`lenskit.util.rng`).
        folds : int
            number of folds of the inner split
        filer : Filer
            filer to manage LensKit-Auto output

        Returns
        -------
        model : Predictor
            the best suited (untrained) predictor for the train dataset, cs parameters.
        incumbent : dict
            a dictionary containing the algorithm name and hyperparameter configuration of the returned model
   """

    if n_trials is None:
        n_trials = np.inf

    # initialize filer if none is provided
    if filer is None:
        filer = Filer()

    # get pre-defined ConfiguraitonSpace if none is provided
    if cs is None:
        n_users = train['user'].nunique()
        n_items = train['item'].nunique()
        cs = get_default_configuration_space(feedback='explicit', n_users=n_users, n_items=n_items)

    if optimization_strategie == 'bayesian':
        if optimization_strategie == 'bayesian':
            incumbent = bayesian_optimization(train=train,
                                              cs=cs,
                                              feedback='explicit',
                                              optimization_metric=optimization_metric,
                                              time_limit_in_sec=time_limit_in_sec,
                                              number_of_evaluations=n_trials,
                                              random_state=random_state,
                                              folds=folds,
                                              filer=filer)
    elif optimization_strategie == 'random_search':
        incumbent = random_search(cs=cs,
                                  train=train,
                                  n_samples=n_trials,
                                  filer=filer,
                                  optimization_metric=optimization_metric,
                                  minimize_error_metric_val=True,
                                  random_state=random_state,
                                  user_feedback='explicit')
    else:
        raise ValueError('optimization_strategie must be either bayesian or random_search')

    # build model from best model configuration found by SMAC
    model = get_model_from_cs(incumbent, feedback='explicit')
    incumbent = incumbent.get_dictionary()

    # return model and model configuration
    return model, incumbent


def find_best_implicit_configuration(train: pd.DataFrame,
                                     cs: ConfigurationSpace = None,
                                     optimization_strategie: str = 'bayesian',
                                     optimization_metric=ndcg,
                                     time_limit_in_sec: int = 300,
                                     n_trials: int = 100,
                                     random_state=42,
                                     folds: int = 1,
                                     filer: Filer = None) -> Tuple[Recommender, dict]:
    """
        returns the best Recommender found in the defined search time

         the find_best_implicit_configuration method will search the ConfigurationSpace
         for the best Recommender model configuration.
         Depending on the ConfigurationSpace parameter provided by the developer,
         performs three different use-cases.
         1. combined algorithm selection and hyperparameter configuration
         2. combined algorthm selection and hyperparameter configuration for a specific subset
            of algorithms and/or different parameter ranges
         3. hyperparameter selection for a specific algorithm.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and parameter ranges defined.
        optimization_strategie: str
            optimization strategie to use. Either bayesian or random_search
        optimization_metric : function
            LensKit recommender metric to optimize for
        time_limit_in_sec : int
            search time limit.
        n_trials : int
                number of samples to be used for optimization_strategy. Value can not be smaller than 6
        random_state
            The random number generator or seed (see :py:func:`lenskit.util.rng`).
        folds : int
            number of folds of the inner split
        filer : Filer
            filer to manage LensKit-Auto output

        Returns
        -------
        model : Predictor
            the best suited (untrained) predictor for the train dataset, cs parameters.
        incumbent : dict
            a dictionary containing the algorithm name and hyperparameter configuration of the returned model
    """

    # initialize filer if none is provided
    if filer is None:
        filer = Filer()

    if cs is None:
        cs = get_default_configuration_space(feedback='implicit')

    if random_state is None:
        random_state = np.random.RandomState()

    if optimization_strategie == 'bayesian':
        incumbent = bayesian_optimization(train=train,
                                          cs=cs,
                                          feedback='implicit',
                                          time_limit_in_sec=time_limit_in_sec,
                                          number_of_evaluations=n_trials,
                                          optimization_metric=optimization_metric,
                                          random_state=random_state,
                                          folds=folds,
                                          filer=filer)

    elif optimization_strategie == 'random_search':
        incumbent = random_search(cs=cs,
                                  train=train,
                                  filer=filer,
                                  optimization_metric=optimization_metric,
                                  random_state=random_state,
                                  n_samples=n_trials,
                                  minimize_error_metric_val=True,
                                  user_feedback='implicit')
    else:
        raise ValueError('optimization_strategie must be either bayesian or random_search')

    # build model from best model configuration found by SMAC
    model = get_model_from_cs(incumbent, feedback='implicit')
    incumbent = incumbent.get_dictionary()

    # return model and model configuration
    return model, incumbent
