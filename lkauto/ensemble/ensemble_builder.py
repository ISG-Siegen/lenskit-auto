import pandas as pd
import numpy as np
from lenskit.data import Dataset

from lkauto.utils.filer import Filer
from lkauto.utils.get_model_from_cs import get_model_from_cs

from lkauto.ensemble.greedy_ensemble_selection import EnsembleSelection


def build_ensemble(train: Dataset,
                   top_n_runs: pd.DataFrame,
                   filer: Filer,
                   ensemble_size: int,
                   lenskit_metric,
                   maximize_metric: bool):
    config_ids = top_n_runs.sort_values(by='error', ascending=True)['run_id']
    ensemble_y = train.interaction_table(format="pandas")['rating']
    ensemble_x = []
    val_indices = None
    bm_cs_list = []

    for config_id in config_ids:
        # Load predictions file for config id, sort and append to ensemble_X
        bm_pred = filer.get_dataframe_from_csv(path_to_file='eval_runs/{}/predictions.csv'.format(config_id))
        bm_cs = filer.get_dict_from_json_file(path_to_file='eval_runs/{}/config_space.json'.format(config_id))

        bm_cs_list.append(bm_cs)

        # Get Validation indices (overwrite for now but should be the same for each base model)
        bm_pred = bm_pred.sort_values(by=list(bm_pred)[0])
        if (val_indices is not None) and (not val_indices.equals(bm_pred[list(bm_pred)[0]])):
            raise ValueError("Validation Indices are not identical between base models!")
        val_indices = bm_pred[list(bm_pred)[0]]

        # Append predictions to ensemble train X
        ensemble_x.append(np.array(bm_pred[list(bm_pred)[1]]))

    ensemble_y = np.array(ensemble_y.loc[val_indices])

    es = EnsembleSelection(ensemble_size=ensemble_size, lenskit_metric=lenskit_metric, maximize_metric=maximize_metric)
    es.ensemble_fit(ensemble_x, ensemble_y)
    es.base_models = [get_model_from_cs(cs, feedback='explicit') for cs, weight in zip(bm_cs_list, es.weights_) if weight > 0]
    es.old_to_new_idx = {old_i: new_i for new_i, old_i in enumerate([idx for idx, weight in enumerate(es.weights_) if weight > 0])}

    incumbent = {"model": str(es),
                 "top_50_models": list(bm_cs_list),
                 "trajectory": list(es.trajectory_),
                 "ensamble_size": es.ensemble_size,
                 "train_loss": es.train_loss_,
                 "weights": list(es.weights_)}

    return es, incumbent
