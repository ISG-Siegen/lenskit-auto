import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.utils.get_model_from_cs import get_explicit_model_from_cs
from lkauto.explicit.explicit_default_config_space import get_explicit_default_configuration_space


def find_best_explicit_configuration(train,
                                     cs=None,
                                     time_limit_in_sec=2700,
                                     random_state=None,
                                     folds=5,
                                     output_dir=None):

    evaler = ExplicitEvaler(train=train,
                            folds=folds)

    if cs is None:
        cs = get_explicit_default_configuration_space()

    if random_state is None:
        random_state = np.random.RandomState()

    scenario = Scenario({
        'run_obj': 'quality',
        'wallclock_limit': time_limit_in_sec,
        'cs': cs,
        'deterministic': False,
        'abort_on_first_run_crash': False,
        'output_dir': output_dir
    })

    smac = SMAC4HPO(scenario=scenario,
                    rng=random_state,
                    tae_runner=evaler.evaluate_explicit)

    try:
        smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    model = get_explicit_model_from_cs(incumbent)
    return model, incumbent


def find_best_implicit_configuration(train, cs=None, time_limit_in_sec=2700, random_state=None):
    raise NotImplementedError
