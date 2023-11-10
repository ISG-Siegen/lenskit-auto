from typing import List, Any

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from lenskit import topn, batch
import logging

from lkauto.utils.filer import Filer
from lkauto.utils.get_model_from_cs import get_model_from_cs
from lkauto.utils.validation_split import validation_split


class ImplicitEvaler:
    """ImplicitEvaler

            the ImplicitEvaler class handles the evaluation of the optimization tool.
            An Evaluation run consists of training a model and predict the performance on a validation split.

            Attributes
            ----------
            train : pd.DataFrame
                pandas dataset containing the train split.
            optimization_metric: function
                LensKit top-n metric used to evaluate the model
            filer : Filer
                filer to organize the output.
            validation : pd.DataFrame
                pandas dataset containing the validation split.
             random_state :
                 The random number generator or seed (see :py:func:`lenskit.util.rng`).
            split_folds :
                The number of folds of the validation split
            split_strategie :
                The strategie used to split the data. Possible values are 'user_based' and 'row_based'
            split_frac :
                The fraction of the data used for the validation split. If the split_folds value is greater than 1,
                this value is ignored.
            num_recommendations :
                The number of recommendations to be made and evaluated for each user.
            minimize_error_metric_val :
                If True, the error metric is minimized. If False, the error metric is maximized. This parameter needs to be
                set in corelation with the optimization metric.

             Methods
            ----------
            evaluate_explicit(config_space: ConfigurationSpace) -> float
    """

    def __init__(self,
                 train: pd.DataFrame,
                 optimization_metric,
                 filer: Filer,
                 validation=None,
                 random_state=42,
                 split_folds: int = 1,
                 split_strategie: str = 'user_based',
                 split_frac: float = 0.25,
                 num_recommendations: int = 10,
                 minimize_error_metric_val: bool = True,
                 ) -> None:
        self.logger = logging.getLogger('lenskit-auto')
        self.train = train
        self.validation = validation
        self.optimization_metric = optimization_metric
        self.random_state = random_state
        self.split_folds = split_folds
        self.split_strategie = split_strategie
        self.split_frac = split_frac
        self.filer = filer
        self.num_recommendations = num_recommendations
        self.minimize_error_metric_val = minimize_error_metric_val
        self.run_id = 0
        # create validation split
        if self.validation is None:
            self.val_fold_indices = validation_split(data=self.train,
                                                     strategie=self.split_strategie,
                                                     num_folds=self.split_folds,
                                                     frac=self.split_frac,
                                                     random_state=self.random_state)
        else:
            self.val_fold_indices = None

    def evaluate(self, config_space: ConfigurationSpace) -> list[Any] | list[int | Any]:
        """ evaluates model defined in config_space

            The config_space parameter defines a model.
            This model is build, trained and evaluated with the validation split.

            Parameters
            ----------
            config_space : ConfigurationSpace
                configuration space containing information to build a model

            Returns
            ----------
            validation_error : float
                the error of the considered model
        """
        output_path = 'eval_runs/'
        self.run_id += 1
        metric_scores = np.array([])
        validation_data = pd.DataFrame()
        scores = None

        # get model form configuration space
        model = get_model_from_cs(config_space, feedback='implicit')

        # iterate over validation folds
        for fold in range(self.split_folds):
            # get validation split by index
            if self.validation is None:
                validation_train = self.train.loc[self.val_fold_indices[fold]["train"], :]
                validation_test = self.train.loc[self.val_fold_indices[fold]["validation"], :]
            else:
                validation_train = self.train
                validation_test = self.validation

            # fit and recommend from configuration
            model = model.fit(validation_train)
            recs = batch.recommend(algo=model, users=validation_test['user'].unique(), n=self.num_recommendations,
                                   n_jobs=1)

            # create rec list analysis
            rla = topn.RecListAnalysis()

            for metric in self.optimization_metric:
                rla.add_metric(metric)

            # compute scores
            scores = rla.compute(recs, validation_test, include_missing=True)

            # store data
            for metric in self.optimization_metric:
                validation_data = pd.concat([validation_data, recs], axis=0)
                metric_scores = np.append(metric_scores, scores[metric.__name__].mean())
            # metric_scores = np.append(metric_scores, scores[self.optimization_metric.__name__].mean())

        # save validation data
        self.filer.save_validataion_data(config_space=config_space,
                                         predictions=validation_data,
                                         metric_scores=metric_scores,
                                         output_path=output_path,
                                         run_id=self.run_id)

        # store score mean and subtract by 1 to enable SMAC to minimize returned value
        metrics_logging = ''
        validation_error_list = []
        for metric in self.optimization_metric:
            validation_error = scores[metric.__name__].mean()
            validation_error_list.append(validation_error)
            metrics_logging += ' | ' + metric.__name__ + '@{}'.format(self.num_recommendations) + ': ' \
                               + str(validation_error)

        self.logger.info('Run ID: ' + str(self.run_id) + ' | ' + str(config_space.get('algo')) + metrics_logging)
        self.logger.debug(str(config_space))

        if self.minimize_error_metric_val:
            return validation_error_list
        else:
            validation_error_list2 = []
            for validation_error in validation_error_list:
                validation_error_list2.append(1 - validation_error)
            return validation_error_list2
