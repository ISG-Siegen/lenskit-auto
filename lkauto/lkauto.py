import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from lkauto.utils.get_default_configurations import get_default_configurations
from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.utils.get_model_from_cs import get_model_from_cs
from lkauto.implicit.implicit_evaler import ImplicitEvaler
from lkauto.utils.get_default_configuration_space import get_default_configuration_space
from lkauto.utils.filer import Filer
from lenskit.algorithms import Predictor
from lenskit import Recommender


def find_best_explicit_configuration(train: pd.DataFrame,
                                     cs: ConfigurationSpace = None,
                                     time_limit_in_sec: int = 2700,
                                     n_trials: int = None,
                                     initial_configuration: list[Configuration] = None,
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
         The hyperparameter and/or model selection process will be stopped after the time_limit_in_sec or (if set) after
         n_trials. The first one to be reached will stop the optimization.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and parameter ranges defined.
        time_limit_in_sec : int
            optimization search time limit in sec.
        n_trials : int
                number of samples to be used for optimization_strategy. Value can not be smaller than 6
                if no initial configuration is provided.
        initial_configuration: list[Configuration]
                list of configurations that should be evaluated first. This parameter can be used to warmstart
                the optimization process.
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

    # get SMAC output directory
    output_dir = filer.get_smac_output_directory_path()

    # initialize ExplicitEvaler for SMAC evaluations
    evaler = ExplicitEvaler(train=train,
                            folds=folds,
                            filer=filer)

    # get pre-defined ConfiguraitonSpace if none is provided
    if cs is None:
        cs = get_default_configuration_space()

    # set RandomState if none is provided
    if random_state is None:
        random_state = np.random.RandomState()

    # set initial configuraiton
    if initial_configuration is None:
        initial_configuraition = get_default_configurations(cs)

    # define SMAC Scenario for algorithm selection and hyperparameter optimization
    scenario = Scenario({
        'run_obj': 'quality',
        'wallclock_limit': time_limit_in_sec,
        'ta_run_limit': n_trials,
        'cs': cs,
        'deterministic': True,
        'abort_on_first_run_crash': False,
        'output_dir': output_dir
    })

    # define SMAC facade for combined algorithm selection and hyperparameter optimization
    smac = SMAC4HPO(scenario=scenario,
                    rng=random_state,
                    tae_runner=evaler.evaluate_explicit,
                    initial_configurations=initial_configuraition,
                    initial_design=None)

    try:
        # start optimizing
        smac.optimize()
    finally:
        # get best model configuration
        incumbent = smac.solver.incumbent

    # build model from best model configuration found by SMAC
    model = get_model_from_cs(incumbent, feedback='explicit')
    incumbent = incumbent.get_dictionary()

    # return model and model configuration
    return model, incumbent


def find_best_implicit_configuration(train: pd.DataFrame,
                                     cs: ConfigurationSpace = None,
                                     time_limit_in_sec: int = 300,
                                     n_trials: int = None,
                                     initial_configuration: list[Configuration] = None,
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
         The hyperparameter and/or model selection process will be stopped after the time_limit_in_sec or (if set) after
         n_trials. The first one to be reached will stop the optimization.

        Parameters
        ----------
        train : pd.DataFrame
            Pandas Dataframe train split.
        cs : ConfigurationSpace
            ConfigurationSpace with all algorithms and parameter ranges defined.
        time_limit_in_sec : int
            optimization search time limit in sec.
        n_trials : int
            number of samples to be used for optimization_strategy. Value can not be smaller than 6
            if no initial configuration is provided.
        initial_configuration : list[Configuration]
            list of configurations that should be evaluated first. This parameter can be used to warmstart
            the optimization process.
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

    # get SMAC output directory
    output_dir = filer.get_smac_output_directory_path()

    if cs is None:
        cs = get_default_configuration_space()

    if random_state is None:
        random_state = np.random.RandomState()

    # set initial configuraiton
    if initial_configuration is None:
        initial_configuraition = get_default_configurations(cs)

    # initialize ImplicitEvaler for SMAC evaluations
    evaler = ImplicitEvaler(train=train,
                            random_state=random_state,
                            folds=folds,
                            filer=filer)

    # define SMAC Scenario for algorithm selection and hyperparameter optimization
    scenario = Scenario({
        'run_obj': 'quality',
        'wallclock_limit': time_limit_in_sec,
        'ta_run_limit': n_trials,
        'cs': cs,
        'deterministic': True,
        'abort_on_first_run_crash': False,
        'output_dir': output_dir
    })

    # define SMAC facade for combined algorithm selection and hyperparameter optimization
    smac = SMAC4HPO(scenario=scenario,
                    rng=random_state,
                    initial_configurations=initial_configuraition,
                    tae_runner=evaler.evaluate_implicit,
                    initial_design=None)

    try:
        # start optimizing
        smac.optimize()
    finally:
        # get best model configuration
        incumbent = smac.solver.incumbent

    # build model from best model configuration found by SMAC
    model = get_model_from_cs(incumbent, feedback='implicit')
    incumbent = incumbent.get_dictionary()

    # return model and model configuration
    return model, incumbent
