# Code Taken from here with adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py

from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd

from lenskit.data import Dataset, ItemList, ItemListCollection
from lenskit.batch import predict


class EnsembleSelection:
    """An ensemble of selected algorithms

    Fitting an EnsembleSelection generates an ensemble from the the models
    generated during the search process. Can be further used for prediction.

    Parameters
    ----------
    lenskit_metric: metric from lenskit
        The metric used to evaluate the models
    maximize_metric: bool = False
        If the metric is to be optimized or not.
    """

    def __init__(self, ensemble_size: int, lenskit_metric,
                 maximize_metric: bool = False) -> None:

        self.maximize_metric = maximize_metric
        self.ensemble_size = ensemble_size

        self.lenskit_metric = lenskit_metric()

        '''
        if maximize_metric:
            def minimized_metric(y_true, y_pred):
                return -self.lenskit_metric.measure_list(y_pred, y_true)
        else:
            def minimized_metric(y_true, y_pred):
                return self.lenskit_metric.measure_list(y_pred, y_true)
                '''

        self.metric = self.minimized_metric

        # Will be filled later from external
        self.base_models = None

    def minimized_metric(self, y_true, y_pred):
        result = self.lenskit_metric.measure_list(y_pred, y_true)
        return -result if self.maximize_metric else result

    def fit(self, data: Dataset):
        """ Fit base models (we assume the ensemble part, ensemble_fit, was already fitted here or is fitted later)

        Parameters
        ----------
        data: Dataset
            Dataset with columns "user", "item", "rating"
        """
        if self.base_models is None:
            raise ValueError("Base Models is None; we need a list of base models to fit them here!")

        for bm in self.base_models:
            bm.fit(data)

        return self

    def predict(self, x_data: ItemListCollection):
        """
        "user", "item" ItemListCollection
        """
        bm_preds = [predict(bm, x_data) for bm in self.base_models]

        ens_predictions = self.ensemble_predict([np.array(bm_pred.to_df()) for bm_pred in bm_preds])

        predictions = bm_preds[0].to_df().copy()

        predictions["score"] = ens_predictions

        predictions_il = ItemListCollection.from_df(predictions, key="user_id")

        return predictions_il

    def ensemble_fit(self, predictions: List[np.ndarray], labels: np.ndarray):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')

        self._fast(predictions, labels)
        self._apply_use_best()
        self._calculate_weights()

        return self

    def _fast(self, predictions: List[np.ndarray], labels: np.ndarray) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )
        for i in range(ensemble_size):
            losses = np.zeros(
                (len(predictions)),
                dtype=np.float64,
            )
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # fant_ensemble_prediction is the prediction of the current ensemble
                # and should be ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
                # We overwrite the contents of fant_ensemble_prediction
                # directly with weighted_ensemble_prediction + new_prediction and then scale for avg
                np.add(
                    weighted_ensemble_prediction,
                    pred,
                    out=fant_ensemble_prediction
                )
                np.multiply(
                    fant_ensemble_prediction,
                    (1. / float(s + 1)),
                    out=fant_ensemble_prediction
                )

                labels_df = pd.DataFrame(labels)
                labels_df.columns = ["rating"]
                labels_df.insert(0, "item_id", labels_df.index)

                fant_ensemble_prediction_df = pd.DataFrame(fant_ensemble_prediction)
                fant_ensemble_prediction_df.columns = ["score"]
                fant_ensemble_prediction_df.insert(0, "item_id", fant_ensemble_prediction_df.index)

                labels_il = ItemList.from_df(labels_df)
                fant_ensemble_prediction_il = ItemList.from_df(fant_ensemble_prediction_df)

                losses[j] = self.metric(labels_il, fant_ensemble_prediction_il)

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
            best = all_best[0]

            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_loss_ = trajectory[-1]

    def _apply_use_best(self):
        # Basically from autogluon the code
        min_score = np.min(self.trajectory_)
        idx_best = self.trajectory_.index(min_score)
        self.indices_ = self.indices_[:idx_best + 1]
        self.trajectory_ = self.trajectory_[:idx_best + 1]
        self.ensemble_size = idx_best + 1
        self.train_loss_ = self.trajectory_[idx_best]

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def ensemble_predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        average = np.zeros_like(predictions[0])
        tmp_predictions = np.empty_like(predictions[0])
        average = average[:, 2]
        tmp_predictions = tmp_predictions[:, 2]

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(self.weights_):
            for pred, weight in zip(predictions, self.weights_):
                # The second column of the pred-array is the score prediction
                # multiply that column with the weights
                # pred[:, 2] = pred[:, 2] * weight
                pred = pred[:, 2]
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)
        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                # The second column of the pred-array is the score prediction
                # multiply that column with the weights
                # pred[:, 2] = pred[:, 2] * weight
                pred = pred[:, 2]
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average
