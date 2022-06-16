import numpy as np
import pandas as pd
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.utils.get_model_from_cs import get_explicit_model_from_cs
from lkauto.explicit.explicit_default_config_space import get_explicit_default_configuration_space
from lkauto.utils.filer import Filer
from lenskit.metrics.predict import rmse
from lkauto.ensembles.ensemble_selection import EnsembleSelection


def build_ensemble(train: pd.DataFrame,
                   evaler: ExplicitEvaler,
                   filer: Filer,
                   ensemble_size: int,
                   lenskit_metric,
                   maximize_metric: bool):
    config_ids = evaler.top_50_runs.sort_values(by='error', ascending=True)['run_id']
    ensemble_y = train['rating']
    ensemble_X = []
    val_indices = None
    bm_cs_list = []

    for config_id in config_ids:
        # Load predictions file for config id, sort and append to ensemble_X
        bm_pred = filer.get_dataframe_from_csv(path_to_file='smac_runs/{}/predictions.csv'.format(config_id))
        bm_cs = filer.get_dict_from_json_file(path_to_file='smac_runs/{}/config_space.json'.format(config_id))

        bm_cs_list.append(bm_cs)

        # Get Validation indices (overwrite for now but should be the same for each base model)
        bm_pred = bm_pred.sort_values(by=list(bm_pred)[0])
        val_indices = bm_pred[list(bm_pred)[0]]

        # Append predictions to ensemble train X
        ensemble_X.append(np.array(bm_pred[list(bm_pred)[1]]))

    ensemble_y = np.array(ensemble_y.loc[val_indices])

    es = EnsembleSelection(ensemble_size=ensemble_size, lenskit_metric=lenskit_metric, maximize_metric=maximize_metric)
    es.ensemble_fit(ensemble_X, ensemble_y)
    es.base_models = [get_explicit_model_from_cs(cs) for cs, weight in zip(bm_cs_list, es.weights_) if weight > 0]
    es.old_to_new_idx = {old_i: new_i for new_i, old_i in enumerate([idx for idx, weight in enumerate(es.weights_) if weight > 0])}

    incumbent = {"model": str(es),
                 "top_50_models": list(bm_cs_list),
                 "trajectory": list(es.trajectory_),
                 "ensamble_size": es.ensemble_size,
                 "train_loss": es.train_loss_,
                 "weights": list(es.weights_)}

    return es, incumbent


def find_best_explicit_configuration(train,
                                     cs=None,
                                     time_limit_in_sec=2700,
                                     random_state=None,
                                     folds=5,
                                     filer=None,
                                     ensemble_size=1,
                                     lenskit_metric=rmse,
                                     maximize_metric=False):

    if filer is None:
        filer = Filer()

    output_dir = filer.get_smac_output_directory_path()

    evaler = ExplicitEvaler(train=train,
                            folds=folds,
                            filer=filer)

    if cs is None:
        cs = get_explicit_default_configuration_space()

    if random_state is None:
        random_state = np.random.RandomState()

    scenario = Scenario({
        'run_obj': 'quality',
        'wallclock_limit': time_limit_in_sec,
        'cs': cs,
        'deterministic': True,
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

    filer.save_dataframe_as_csv(evaler.top_50_runs, '', 'top_50_runs')

    if ensemble_size > 1:
        model, incumbent = build_ensemble(train, evaler, filer, ensemble_size, lenskit_metric, maximize_metric)
    else:
        model = get_explicit_model_from_cs(incumbent)
        incumbent.get_dictionary()

    return model, incumbent


def find_best_implicit_configuration(train, cs=None, time_limit_in_sec=2700, random_state=None):
    raise NotImplementedError
