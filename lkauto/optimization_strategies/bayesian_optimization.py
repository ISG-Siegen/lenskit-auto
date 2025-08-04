import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from lenskit import Pipeline

# from smac.facade.smac_hpo_facade import SMAC4HPO
# from smac.scenario.scenario import Scenario

from smac import HyperparameterOptimizationFacade
from smac.initial_design import RandomInitialDesign
from smac.intensifier import Intensifier
from smac.scenario import Scenario

from lenskit.data import Dataset, ItemListCollection

from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.implicit.implicit_evaler import ImplicitEvaler
from lkauto.utils.filer import Filer
from lkauto.utils.get_default_configuration_space import get_default_configuration_space

from typing import Tuple, Optional
import logging


def bayesian_optimization(train: Dataset,
                          user_feedback: str,
                          validation: ItemListCollection = None,
                          cs: ConfigurationSpace = None,
                          optimization_metric=None,
                          time_limit_in_sec: int = 2700,
                          num_evaluations: int = 100,
                          random_state=None,
                          split_folds: int = 1,
                          split_strategie: str = 'user_based',
                          split_frac: float = 0.25,
                          ensemble_size: int = 50,
                          num_recommendations: int = 10,
                          minimize_error_metric_val: bool = True,
                          predict_mode: bool = True,
                          filer: Filer = None) -> Tuple[Configuration, Optional[Pipeline], Optional[pd.DataFrame]]:
    """
        returns the best configuration found by bayesian optimization.
        The bayesian_optimization method will use SMAC3 to find the best
        performing configuration for the given train split.
        The ConfigurationSpace can consist of hyperparameters for a
        single algorithm or a combination of algorithms.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe outer train split.
        validation : pd.DataFrame
            Pandas Dataframe validation split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and hyperparameter ranges defined.
        time_limit_in_sec : int
            time limit in seconds for the optimization process
        num_evaluations : int
            number of samples to be drawn from the ConfigurationSpace
        optimization_metric : function
            LensKit prediction accuracy metric to optimize for.
        minimize_error_metric_val : bool
            Bool that decides if the error metric should be minimized or maximized.
        predict_mode : bool
            If set to true, indicates that a prediction model should be created, a recommender model otherwise
        user_feedback : str
            Defines if the dataset contains explicit or implicit feedback.
        random_state: int
        split_folds : int
            number of folds for cross validation
        split_strategie : str
            cross validation strategie (user_based or item_based)
        split_frac : float
            fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
            will be ignored.
        ensemble_size : int
            number of models to be used in the ensemble of rating prediction tasks. This value will be ignored
            for recommender tasks.
        num_recommendations : int
            number of recommendations to be made for each user. This value will be ignored for rating prediction.
        filer : Filer
            filer to manage LensKit-Auto output

        Returns
        -------
        incumbent : Configuration
            best configuration found by bayesian optimization
        top_n_runs : pd.DataFrame
            top n runs found by bayesian optimization
    """
    logger = logging.getLogger('lenskit-auto')
    logger.info('--Start Bayesian Optimization--')

    # get SMAC output directory
    output_dir = filer.get_smac_output_directory_path()

    # initialize Evaler for SMAC evaluations
    if user_feedback == 'explicit':
        evaler = ExplicitEvaler(train=train,
                                validation=validation,
                                optimization_metric=optimization_metric,
                                filer=filer,
                                random_state=random_state,
                                split_folds=split_folds,
                                split_strategy=split_strategie,
                                split_frac=split_frac,
                                ensemble_size=ensemble_size,
                                minimize_error_metric_val=minimize_error_metric_val,
                                predict_mode=predict_mode)
    elif user_feedback == 'implicit':
        evaler = ImplicitEvaler(train=train,
                                validation=validation,
                                optimization_metric=optimization_metric,
                                filer=filer,
                                random_state=random_state,
                                split_folds=split_folds,
                                split_strategy=split_strategie,
                                split_frac=split_frac,
                                num_recommendations=num_recommendations,
                                minimize_error_metric_val=minimize_error_metric_val,
                                predict_mode=predict_mode)
    else:
        raise ValueError('feedback must be either explicit or implicit')

    # get pre-defined ConfiguraitonSpace if none is provided
    if cs is None:
        logger.debug('initializing default ConfigurationSpace')
        cs = get_default_configuration_space(data=train,
                                             validation=validation,
                                             feedback='explicit',
                                             random_state=random_state)

    # initial_configuraition = get_default_configurations(cs)

    scenario = Scenario(configspace=cs,
                        deterministic=True,
                        n_trials=num_evaluations,
                        walltime_limit=time_limit_in_sec,
                        output_directory=output_dir
                        )
    # Target function that should minimize the error of th evaler
    # Also saves the model from the smallest error evaluation
    best_error = float('inf')
    best_model = None

    def target_function(config, seed=0, budget=None, instance=None):
        error, model = evaler.evaluate(config)

        nonlocal best_error, best_model

        if error < best_error:
            best_error = error
            best_model = model

        return error

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=target_function,
        initial_design=RandomInitialDesign(scenario),
        intensifier=Intensifier(scenario),
        overwrite=True
    )

    incumbent = smac.optimize()

    logger.info('--End Bayesian Optimization--')

    # return best model configuration
    if user_feedback == 'explicit':
        # save top n runs
        filer.save_dataframe_as_csv(evaler.top_n_runs, '', 'top_n_runs')
        # filer.save_model(model=incumbent.)
        return incumbent, best_model, evaler.top_n_runs
    elif user_feedback == 'implicit':
        return incumbent, best_model, None
    else:
        raise ValueError('feedback must be either explicit or implicit')
