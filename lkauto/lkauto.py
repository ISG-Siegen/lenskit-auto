import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.utils.get_model_from_cs import get_explicit_model_from_cs
from lkauto.explicit.explicit_default_config_space import get_explicit_default_configuration_space
from lkauto.utils.filer import Filer
from lenskit.algorithms import Predictor


def find_best_explicit_configuration(train: pd.DataFrame,
                                     cs: ConfigurationSpace = None,
                                     time_limit_in_sec: int = 2700,
                                     random_state=None,
                                     folds: int = 5,
                                     filer: Filer = None) -> tuple[Predictor, dict]:
    """ returns the best Predictor found in the defined search time

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

    # initialize ExplicitEvaler for SMAC evaluations
    evaler = ExplicitEvaler(train=train,
                            folds=folds,
                            filer=filer)

    # get pre-defined ConfiguraitonSpace if none is provided
    if cs is None:
        cs = get_explicit_default_configuration_space()

    # set RandomState if none is provided
    if random_state is None:
        random_state = np.random.RandomState()

    # define SMAC Scenario for algorithm selection and hyperparameter optimization
    scenario = Scenario({
        'run_obj': 'quality',
        'wallclock_limit': time_limit_in_sec,
        'cs': cs,
        'deterministic': True,
        'abort_on_first_run_crash': False,
        'output_dir': output_dir
    })

    # define SMAC facade for combined algorithm selection and hyperparameter optimization
    smac = SMAC4HPO(scenario=scenario,
                    rng=random_state,
                    tae_runner=evaler.evaluate_explicit)

    try:
        # start optimizing
        smac.optimize()
    finally:
        # get best model configuration
        incumbent = smac.solver.incumbent

    # build model from best model configuration found by SMAC
    model = get_explicit_model_from_cs(incumbent)
    incumbent = incumbent.get_dictionary()

    # return model and model configuration
    return model, incumbent

def find_best_implicit_configuration(train, cs=None, time_limit_in_sec=2700, random_state=None):
    raise NotImplementedError
