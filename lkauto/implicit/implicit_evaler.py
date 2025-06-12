import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace

from lenskit.batch import predict
from lenskit import batch
from lenskit.data import Dataset
from lenskit.metrics import RunAnalysis
from lenskit.splitting import TTSplit
import logging

from lenskit.pipeline import predict_pipeline

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
            split_strategy :
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
                 train: Dataset,
                 optimization_metric,
                 filer: Filer,
                 validation=None,
                 random_state=42,
                 split_folds: int = 1,
                 split_strategy: str = 'user_based',
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
        self.split_strategy = split_strategy
        self.split_frac = split_frac
        self.filer = filer
        self.num_recommendations = num_recommendations
        self.minimize_error_metric_val = minimize_error_metric_val
        self.run_id = 0
        # create validation split
        if self.validation is None:
            self.val_fold_indices = validation_split(data=self.train,
                                                     strategy=self.split_strategy,
                                                     num_folds=self.split_folds,
                                                     frac=self.split_frac,
                                                     random_state=self.random_state)
        else:
            self.val_fold_indices = None

    def evaluate(self, config_space: ConfigurationSpace) -> float:
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

        if self.validation is None:
            for fold in self.val_fold_indices:
                validation_train = fold.train
                validation_test = fold.test

                pipeline = predict_pipeline(scorer=model)
                fit_pipeline = pipeline.clone()
                fit_pipeline.train(validation_train)

                recs = predict(fit_pipeline, validation_test)

                # create rec list analysis
                rla = RunAnalysis()
                rla.add_metric(self.optimization_metric)

                # compute scores
                scores = rla.measure(recs, validation_test)

                # store data
                validation_data = pd.concat([validation_data, recs.to_df()], axis=0)
                # the first (index 0) column should contain the means for the metrics (rows)
                metric_scores = np.append(metric_scores, scores.list_summary().loc[self.optimization_metric.__name__, "mean"])
        else:
            for fold in range(self.split_folds):
                validation_train = self.train
                validation_test = self.validation

                pipeline = predict_pipeline(scorer=model)
                fit_pipeline = pipeline.clone()
                fit_pipeline.train(validation_train)

                recs = predict(fit_pipeline, validation_test)

                # create rec list analysis
                rla = RunAnalysis()
                rla.add_metric(self.optimization_metric)

                # compute scores
                scores = rla.measure(recs, validation_test)

                # store data
                validation_data = pd.concat([validation_data, recs.to_df()], axis=0)
                metric_scores = np.append(metric_scores, scores.list_summary().loc[self.optimization_metric.__name__, "mean"])


        # save validation data
        self.filer.save_validataion_data(config_space=config_space,
                                         predictions=validation_data,
                                         metric_scores=metric_scores,
                                         output_path=output_path,
                                         run_id=self.run_id)

        # store score mean and subtract by 1 to enable SMAC to minimize returned value
        # the first (index 0) column should contain the means for the metrics (rows)
        validation_error = scores.list_summary().loc[self.optimization_metric.__name__, "mean"] - 1

        self.logger.info('Run ID: ' + str(self.run_id) + ' | ' + str(config_space.get('algo')) + ' | ' +
                         self.optimization_metric.__name__ + '@{}'.format(self.num_recommendations) + ': '
                         + str(validation_error))
        self.logger.debug(str(config_space))

        if self.minimize_error_metric_val:
            return validation_error
        else:
            return 1 - validation_error
