import pandas as pd

from ConfigSpace import Configuration, ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

from lkauto.utils.get_default_configurations import get_default_configurations
from lkauto.utils.filer import Filer
from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.implicit.implicit_evaler import ImplicitEvaler

from typing import Tuple


def bayesian_optimization(train: pd.DataFrame,
                          feedback: str,
                          cs: ConfigurationSpace = None,
                          optimization_metric=None,
                          time_limit_in_sec: int = 2700,
                          number_of_evaluations: int = 100,
                          random_state=None,
                          folds: int = 5,
                          filer: Filer = None) -> Tuple[Configuration, dict]:
    # get SMAC output directory
    output_dir = filer.get_smac_output_directory_path()

    # initialize Evaler for SMAC evaluations
    if feedback == 'explicit':
        evaler = ExplicitEvaler(train=train,
                                folds=folds,
                                optimization_metric=optimization_metric,
                                filer=filer,
                                random_state=random_state)
    elif feedback == 'implicit':
        evaler = ImplicitEvaler(train=train,
                                optimization_metric=optimization_metric,
                                folds=folds,
                                filer=filer,
                                random_state=random_state)
    else:
        raise ValueError('feedback must be either explicit or implicit')

    # set initial configuraiton
    initial_configuraition = get_default_configurations(cs)

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
                    tae_runner=evaler.evaluate_explicit,
                    initial_configurations=initial_configuraition,
                    initial_design=None)

    try:
        # start optimizing
        smac.optimize()
    finally:
        # get best model configuration
        incumbent = smac.solver.incumbent

    return incumbent
