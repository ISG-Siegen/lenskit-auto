import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace

from lkauto.optimization_strategies.bayesian_optimization import bayesian_optimization
from lkauto.explicit.explicit_default_config_space import get_explicit_default_configuration_space
from lkauto.implicit.implicit_default_config_space import get_implicit_default_configuration_space
from lkauto.optimization_strategies.random_search import random_search
from lkauto.utils.filer import Filer
from lkauto.utils.get_model_from_cs import get_explicit_model_from_cs, get_implicit_recommender_from_cs

from lenskit.algorithms import Predictor
from lenskit import Recommender


def find_best_explicit_configuration(train: pd.DataFrame,
                                     cs: ConfigurationSpace = None,
                                     optimization_strategie: str = 'bayesian',
                                     time_limit_in_sec: int = 2700,
                                     number_of_evaluations: int = 100,
                                     random_state=None,
                                     folds: int = 5,
                                     filer: Filer = None) -> tuple[Predictor, dict]:
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

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and parameter ranges defined.
        time_limit_in_sec : int
            search time limit.
        optimization_strategie: str
            optimization strategie to use. Either bayesian or random_search.
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

    # set RandomState if none is provided
    if random_state is None:
        random_state = np.random.RandomState()

    # get pre-defined ConfiguraitonSpace if none is provided
    if cs is None:
        cs = get_explicit_default_configuration_space()

    if optimization_strategie == 'bayesian':
        incumbent = bayesian_optimization(train=train,
                                          cs=cs,
                                          feedback='explicit',
                                          time_limit_in_sec=time_limit_in_sec,
                                          number_of_evaluations=number_of_evaluations,
                                          random_state=random_state,
                                          folds=folds,
                                          filer=filer)

    elif optimization_strategie == 'random_search':
        incumbent = random_search(cs=cs,
                                  train=train,
                                  n_samples=number_of_evaluations,
                                  minimize_error_metric_val=True,
                                  random_state=random_state,
                                  user_feedback='explicit')
    else:
        raise ValueError('optimization_strategie must be either bayesian or random_search')

    # build model from best model configuration found by SMAC
    model = get_explicit_model_from_cs(incumbent)
    incumbent = incumbent.get_dictionary()

    # return model and model configuration
    return model, incumbent


def find_best_implicit_configuration(train: pd.DataFrame,
                                     cs: ConfigurationSpace = None,
                                     optimization_strategie: str = 'bayesian',
                                     time_limit_in_sec: int = 300,
                                     number_of_evaluations: int = 100,
                                     random_state=None,
                                     folds: int = 1,
                                     filer: Filer = None) -> tuple[Recommender, dict]:
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
        time_limit_in_sec : int
            search time limit.
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
        cs = get_implicit_default_configuration_space()

    if random_state is None:
        random_state = np.random.RandomState()

    if optimization_strategie == 'bayesian':
        incumbent = bayesian_optimization(train=train,
                                          cs=cs,
                                          feedback='implicit',
                                          time_limit_in_sec=time_limit_in_sec,
                                          number_of_evaluations=number_of_evaluations,
                                          random_state=random_state,
                                          folds=folds,
                                          filer=filer)

    elif optimization_strategie == 'random_search':
        incumbent = random_search(cs=cs,
                                  train=train,
                                  n_samples=number_of_evaluations,
                                  minimize_error_metric_val=True,
                                  user_feedback='implicit')
    else:
        raise ValueError('optimization_strategie must be either bayesian or random_search')

    # build model from best model configuration found by SMAC
    model = get_implicit_recommender_from_cs(incumbent)
    incumbent = incumbent.get_dictionary()

    # return model and model configuration
    return model, incumbent
