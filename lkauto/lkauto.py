import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace

from lkauto.utils.get_model_from_cs import get_model_from_cs
from lkauto.optimization_strategies.bayesian_optimization import bayesian_optimization
from lkauto.optimization_strategies.random_search import random_search
from lkauto.utils.filer import Filer
from lkauto.ensemble.ensemble_builder import build_ensemble
from lkauto.preprocessing.preprocessing import preprocess_data
from lkauto.utils.logging import get_logger

# from lenskit.metrics.predict import rmse
# from lenskit.metrics.topn import ndcg
# from lenskit.algorithms import Predictor
# from lenskit import Recommender

from lenskit.metrics.predict import RMSE
from lenskit.metrics import NDCG
from lenskit.pipeline import Component
from lenskit.data import Dataset, ItemListCollection

from typing import Tuple


def get_best_prediction_model(train: Dataset,
                              validation: ItemListCollection = None,
                              cs: ConfigurationSpace = None,
                              optimization_metric=RMSE,
                              optimization_strategie: str = 'bayesian',
                              time_limit_in_sec: int = 2700,
                              num_evaluations: int = 500,
                              random_state=None,
                              split_folds: int = 1,
                              split_frac: float = 0.25,
                              split_strategie: str = 'user_based',
                              ensemble_size: int = 50,
                              minimize_error_metric_val: bool = True,
                              min_number_of_ratings: int = None,
                              max_number_of_ratings: int = None,
                              drop_duplicates: bool = False,
                              drop_na_values: bool = False,
                              user_column: str = 'user',
                              item_column: str = 'item',
                              rating_column: str = 'rating',
                              timestamp_col: str = 'timestamp',
                              include_timestamp: bool = True,
                              log_level: str = 'INFO',
                              filer: Filer = None) -> Tuple[Component, dict]:
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
        validation : pd.DataFrame
            Pandas Dataframe validation split.
            if a validation split is provided, split_folds, split_strategy and split_frac will be ignored.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and parameter ranges defined.
        optimization_metric : function
            LensKit prediction accuracy metric to optimize for (either rmse or mae)
        optimization_strategie: str
            optimization strategie to use. Either bayesian or random_search
        time_limit_in_sec : int
            optimization search time limit in sec.
        num_evaluations : int
                number of samples to be used for optimization_strategy. Value can not be smaller than 6
                if no initial configuration is provided.
        random_state
            The random number generator or seed (see :py:func:`lenskit.util.rng`).
        split_folds : int
            number of folds of the inner split
        split_frac : float
            fraction of the inner split. If split_folds is to a value > 2, split_frac will be ignored.
            Value must be between 0 and 1.
        split_strategie : str
            split strategie to use. Either 'user_based' or 'item_based'
        ensemble_size : int
            number of models to be used for ensemble building.
        minimize_error_metric_val : bool
            if True, the optimization will try to minimize the error metric value.
            If False, the optimization will try to maximize the error metric value.
        min_number_of_ratings : int
            minimum number of ratings a user must have to be considered in the train dataset.
        max_number_of_ratings : int
            maximum number of ratings a user can have to be considered in the train dataset.
        drop_duplicates : bool
            if True, all duplicate rows will be dropped from the train dataset.
        drop_na_values : bool
            if True, all rows with NaN values will be dropped from the train dataset.
        user_column : str
            name of the user column in the train dataset.
        item_column : str
            name of the item column in the train dataset.
        rating_column : str
            name of the rating column in the train dataset.
        timestamp_col: str
            Name of the timestamp column
        include_timestamp: bool = True
            If True, the timestamp column will be included in the dataset
        log_level : str
            log level to use.
        filer : Filer
            filer to manage LensKit-Auto output

        Returns
        -------
        model : Predictor
            the best suited (untrained) predictor for the train dataset, cs parameters.
        incumbent : dict
            a dictionary containing the algorithm name and hyperparameter configuration of the returned model
   """

    logger = get_logger(level=log_level)

    # log parameters
    logger.info('---Starting LensKit-Auto---')
    if time_limit_in_sec is not None:
        logger.info('\t optimization_time: \t\t {} seconds'.format(time_limit_in_sec))
    if num_evaluations is not None:
        logger.info('\t num_evaluations: \t\t\t {}'.format(num_evaluations))
    logger.info('\t optimization_metric: \t\t {}'.format(optimization_metric.__name__))
    logger.info('\t optimization_strategie: \t {}'.format(optimization_strategie))

    # set num_evaluations to infinity if none is provided
    if num_evaluations is None:
        logger.debug('setting up num_evaluations to infinity')
        num_evaluations = np.inf

    # initialize filer if none is provided
    if filer is None:
        logger.debug('initializing filer')
        filer = Filer()

    # set random_state if none is provided
    if random_state is None:
        logger.debug('initializing random_state')
        random_state = 42

    # set split_folds to 1 if validation is not None
    if validation is not None:
        split_folds = 1

    # preprocess data
    preprocess_data(data=train,
                    user_col=user_column,
                    item_col=item_column,
                    rating_col=rating_column,
                    timestamp_col=timestamp_col,
                    include_timestamp=include_timestamp,
                    min_interactions_per_user=min_number_of_ratings,
                    max_interactions_per_user=max_number_of_ratings,
                    drop_na_values=drop_na_values,
                    drop_duplicates=drop_duplicates)

    # decide which optimization strategy to use
    if optimization_strategie == 'bayesian':
        incumbent, top_n_runs = bayesian_optimization(train=train,
                                                      cs=cs,
                                                      user_feedback='explicit',
                                                      validation=validation,
                                                      optimization_metric=optimization_metric,
                                                      time_limit_in_sec=time_limit_in_sec,
                                                      num_evaluations=num_evaluations,
                                                      random_state=random_state,
                                                      split_folds=split_folds,
                                                      split_frac=split_frac,
                                                      split_strategie=split_strategie,
                                                      ensemble_size=ensemble_size,
                                                      minimize_error_metric_val=minimize_error_metric_val,
                                                      filer=filer)
    elif optimization_strategie == 'random_search':
        incumbent, top_n_runs = random_search(train=train,
                                              cs=cs,
                                              user_feedback='explicit',
                                              validation=validation,
                                              optimization_metric=optimization_metric,
                                              time_limit_in_sec=time_limit_in_sec,
                                              num_evaluations=num_evaluations,
                                              random_state=random_state,
                                              split_folds=split_folds,
                                              split_frac=split_frac,
                                              split_strategie=split_strategie,
                                              ensemble_size=ensemble_size,
                                              minimize_error_metric_val=minimize_error_metric_val,
                                              filer=filer)
    else:
        raise ValueError('optimization_strategie must be either bayesian or random_search')

    # save top_n_runs to csv
    filer.save_dataframe_as_csv(top_n_runs, '', 'top_n_runs')

    logger.info('--Start Postrprocessing--')
    if ensemble_size > 1:
        model, incumbent = build_ensemble(train=train, top_n_runs=top_n_runs,
                                          filer=filer,
                                          ensemble_size=ensemble_size,
                                          lenskit_metric=optimization_metric,
                                          maximize_metric=(not minimize_error_metric_val))
        logger.info('--Best Model--')
        logger.info('GES Ensemble Model')
    else:
        # build model from best model configuration found by SMAC
        model = get_model_from_cs(incumbent, feedback='explicit')
        incumbent = incumbent.get_dictionary()
        logger.info('--Best Model--')
        logger.info(incumbent)

    logger.info('---LensKit-Auto finished---')

    # return model and model configuration
    return model, incumbent


def get_best_recommender_model(train: Dataset,
                               validation: ItemListCollection = None,
                               cs: ConfigurationSpace = None,
                               optimization_metric=NDCG,
                               optimization_strategie: str = 'bayesian',
                               time_limit_in_sec: int = 90,
                               num_evaluations: int = 500,
                               random_state=None,
                               split_folds: int = 1,
                               split_frac: float = 0.25,
                               split_strategie: str = 'user_based',
                               minimize_error_metric_val: bool = False,
                               num_recommendations: int = 10,
                               min_interactions_per_user: int = None,
                               max_interactions_per_user: int = None,
                               drop_duplicates: bool = False,
                               drop_na_values: bool = False,
                               user_column: str = 'user',
                               item_column: str = 'item',
                               rating_column: str = 'rating',
                               timestamp_col: str = 'timestamp',
                               include_timestamp: bool = True,
                               log_level: str = 'INFO',
                               filer: Filer = None) -> Tuple[Component, dict]:
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
         The hyperparameter and/or model selection process will be stopped after the time_limit_in_sec or (if set) after
         n_trials. The first one to be reached will stop the optimization.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        validation : pd.DataFrame
            Pandas Dataframe validation split.
            if a validation split is provided, split_folds, split_strategy and split_frac will be ignored.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and parameter ranges defined.
        optimization_strategie: str
            optimization strategie to use. Either bayesian or random_search
        optimization_metric : function
            LensKit recommender metric to optimize for
        time_limit_in_sec : int
            search time limit.
        num_evaluations : int
                number of samples to be used for optimization_strategy. Value can not be smaller than 6
        random_state
            The random number generator or seed (see :py:func:`lenskit.util.rng`).
        split_folds : int
            number of folds of the inner split
        split_frac : float
            fraction of the inner split. If split_folds is not None, split_frac will be ignored.
            Value must be between 0 and 1.
        split_strategie : str
            split strategie to use. Either 'user_based' or 'item_based'.
        minimize_error_metric_val : bool
            if True, the optimization_metric value will be minimized.
            If False, the optimization_metric value will be maximized.
        num_recommendations : int
            number of recommendations to be evaluted per user. Value must be greater than 0.
        min_interactions_per_user : int
            minimum number of ratings a user must have to be considered in the train dataset.
        max_interactions_per_user : int
            maximum number of ratings a user can have to be considered in the train dataset.
        drop_duplicates : bool
            if True, all duplicate rows will be dropped from the train dataset.
        drop_na_values : bool
            if True, all rows with NaN values will be dropped from the train dataset.
        user_column : str
            name of the user column in the train dataset.
        item_column : str
            name of the item column in the train dataset.
        rating_column : str
            name of the rating column in the train dataset.
        timestamp_col: str
            Name of the timestamp column
        include_timestamp: bool = True
            If True, the timestamp column will be included in the dataset
        log_level : str
            log level to use.
        filer : Filer
            filer to manage LensKit-Auto output

        Returns
        -------
        model : Predictor
            the best suited (untrained) predictor for the train dataset, cs parameters.
        incumbent : dict
            a dictionary containing the algorithm name and hyperparameter configuration of the returned model
    """

    logger = get_logger(level=log_level)

    # log parameters
    logger.info('---Starting LensKit-Auto---')
    if time_limit_in_sec is not None:
        logger.info('\t optimization_time: \t\t {} seconds'.format(time_limit_in_sec))
    if num_evaluations is not None:
        logger.info('\t num_evaluations: \t\t\t {}'.format(num_evaluations))
    logger.info('\t optimization_metric: \t\t {}@{}'.format(optimization_metric.__name__, num_recommendations))
    logger.info('\t optimization_strategie: \t {}'.format(optimization_strategie))

    # set num_evaluations to infinity if none is provided
    if num_evaluations is None:
        logger.debug('num_evaluations is None. Setting num_evaluations to infinity.')
        num_evaluations = np.inf

    # initialize filer if none is provided
    if filer is None:
        logger.debug('filer is None. Initializing filer.')
        filer = Filer()

    # set random state if none is provided
    if random_state is None:
        logger.debug('random_state is None. Initializing random_state.')
        random_state = 42

    # set split_folds to 1 if validation is not None
    if validation is not None:
        split_folds = 1

    # preprocess data
    preprocess_data(data=train,
                    user_col=user_column,
                    item_col=item_column,
                    rating_col=rating_column,
                    timestamp_col=timestamp_col,
                    include_timestamp=include_timestamp,
                    min_interactions_per_user=min_interactions_per_user,
                    max_interactions_per_user=max_interactions_per_user,
                    drop_na_values=drop_na_values,
                    drop_duplicates=drop_duplicates)

    # define optimization strategie to use
    if optimization_strategie == 'bayesian':
        incumbent = bayesian_optimization(train=train,
                                          validation=validation,
                                          cs=cs,
                                          user_feedback='implicit',
                                          optimization_metric=optimization_metric,
                                          time_limit_in_sec=time_limit_in_sec,
                                          num_evaluations=num_evaluations,
                                          random_state=random_state,
                                          split_folds=split_folds,
                                          split_frac=split_frac,
                                          split_strategie=split_strategie,
                                          minimize_error_metric_val=minimize_error_metric_val,
                                          num_recommendations=num_recommendations,
                                          filer=filer)
    elif optimization_strategie == 'random_search':
        incumbent = random_search(train=train,
                                  validation=validation,
                                  cs=cs,
                                  user_feedback='implicit',
                                  optimization_metric=optimization_metric,
                                  time_limit_in_sec=time_limit_in_sec,
                                  num_evaluations=num_evaluations,
                                  random_state=random_state,
                                  split_folds=split_folds,
                                  split_frac=split_frac,
                                  split_strategie=split_strategie,
                                  minimize_error_metric_val=minimize_error_metric_val,
                                  num_recommendations=num_recommendations,
                                  filer=filer)
    else:
        raise ValueError('optimization_strategie must be either bayesian or random_search')

    logger.info('--Start Postrprocessing--')

    # build model from best model configuration found by SMAC
    model = get_model_from_cs(incumbent, feedback='implicit')
    incumbent = incumbent.get_dictionary()

    logger.info('--Best Model--')
    logger.info(incumbent)

    logger.info('---LensKit-Auto finished---')

    # return model and model configuration
    return model, incumbent
