import logging

import numpy as np
import pandas as pd

from typing import Tuple

from lenskit.data import Dataset, ItemListCollection
from lenskit.pipeline import Pipeline, predict_pipeline, topn_pipeline
from lenskit.batch import predict, recommend
from lenskit.metrics import RunAnalysis
from ConfigSpace import ConfigurationSpace

from lkauto.utils.filer import Filer
from lkauto.utils.get_model_from_cs import get_model_from_cs
from lkauto.utils.validation_split import validation_split
from lkauto.utils.update_top_n_runs import update_top_n_runs


class ExplicitEvaler:
    """ExplicitEvaler

        the ExplicitEvaler class handles the evaluation of rating prediction models.
        An Evaluation run consists of training a model and to predict and evaluate
        the performance on a validation split.

        Attributes
        ----------
        train : Dataset
            lenskit dataset containing the train split.
        optimization_metric: function
            LensKit prediction accuracy metric used to evaluate the model (either rmse or mae)
        filer : Filer
            Filer to organize the output.
        validation : ItemListCollection
            An ItemListCollection containing the validation split.
        random_state :
            The random number generator or seed (see :py:func:`lenskit.util.rng`).
        split_folds :
            The number of folds of the validation split
        split_strategy :
            The strategy used to split the data (either "user_based" or "row_based")
        split_frac :
            The fraction of the data used for the validation split. If the split_folds value is greater than 1,
            this value is ignored.
        ensemble_size :
            The number of models used to build the final ensemble predictor.
        minimize_error_metric_val :
            If True, the error metric is minimized. If False, the error metric is maximized. This parameter needs to be
            set in corelation with the optimization metric.
        predict_mode : bool
            If set to true, indicates that a prediction model should be created, a recommender model otherwise

        Methods
        ----------
        evaluate_explicit(config_space: ConfigurationSpace) -> float
    """

    def __init__(self,
                 train: Dataset,
                 optimization_metric,
                 filer: Filer,
                 validation: ItemListCollection = None,
                 random_state=42,
                 split_folds: int = 1,
                 split_strategy: str = 'user_based',
                 split_frac: float = 0.25,
                 ensemble_size: int = 50,
                 minimize_error_metric_val: bool = True,
                 predict_mode: bool = True
                 ) -> None:
        self.logger = logging.getLogger('lenskit-auto')
        self.train = train
        self.filer = filer
        self.validation = validation
        self.random_state = random_state
        self.split_folds = split_folds
        self.optimization_metric = optimization_metric
        self.split_strategy = split_strategy
        self.split_frac = split_frac
        self.minimize_error_metric_val = minimize_error_metric_val
        self.predict_mode = predict_mode
        self.run_id = 0
        self.ensemble_size = ensemble_size
        self.top_n_runs = pd.DataFrame(columns=['run_id', 'model', 'error'])
        if self.validation is None:
            self.train_test_splits = list(validation_split(data=self.train,
                                                      strategy=self.split_strategy,
                                                      num_folds=self.split_folds,
                                                      frac=self.split_frac,
                                                      random_state=self.random_state))
        else:
            self.train_test_splits = None

    def evaluate(self, config_space: ConfigurationSpace) -> Tuple[float, Pipeline]:
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
        error_metric = np.array([])
        validation_data = pd.DataFrame()

        best_mean = float('inf')
        best_model = None

        # get model from configuration space
        model = get_model_from_cs(config_space, feedback='explicit')

        if self.validation is None:
            for fold in self.train_test_splits:
                validation_train = fold.train
                validation_test = fold.test

                # initialize pipeline
                pipeline = None
                if self.predict_mode:
                    pipeline = predict_pipeline(scorer=model)
                else:
                    pipeline = topn_pipeline(scorer=model)

                fit_pipeline = pipeline.clone()
                fit_pipeline.train(data=validation_train)

                recs = None
                if self.predict_mode:
                    recs = predict(fit_pipeline, validation_test)
                else:
                    recs = recommend(fit_pipeline, validation_test)

                run_analysis = RunAnalysis()
                run_analysis.add_metric(self.optimization_metric)
                error_results = run_analysis.measure(recs, validation_test)

                # if error is smaller than before, save model
                error_results_mean = error_results.list_summary().loc[self.optimization_metric.__name__, "mean"]
                if error_results_mean < best_mean:
                    best_mean = error_results_mean
                    best_model = fit_pipeline

                error_metric = np.append(error_metric, error_results_mean)
                validation_data = pd.concat([validation_data, recs.to_df()], ignore_index=True)
        else:
            for fold in range(self.split_folds):
                validation_train = self.train
                validation_test = self.validation

                # initialize pipeline
                pipeline = None
                if self.predict_mode:
                    pipeline = predict_pipeline(scorer=model)
                else:
                    pipeline = topn_pipeline(scorer=model)

                fit_pipeline = pipeline.clone()
                fit_pipeline.train(data=validation_train)

                recs = None
                if self.predict_mode:
                    recs = predict(fit_pipeline, validation_test)
                else:
                    recs = recommend(fit_pipeline, validation_test)

                run_analysis = RunAnalysis()
                run_analysis.add_metric(self.optimization_metric)
                error_results = run_analysis.measure(recs, validation_test)

                # if error is smaller than before, save model
                error_results_mean = error_results.list_summary().loc[self.optimization_metric.__name__, "mean"]
                if error_results_mean < best_mean:
                    best_mean = error_results_mean
                    best_model = fit_pipeline

                error_metric = np.append(error_metric, error_results_mean)
                validation_data = pd.concat([validation_data, recs.to_df()], ignore_index=True)

        # Save validation data for reproducibility and ensembling
        self.top_n_runs = update_top_n_runs(config_space=config_space,
                                            errors=error_metric,
                                            num_models=self.ensemble_size,
                                            run_id=self.run_id,
                                            top_n_runs=self.top_n_runs,
                                            pipeline=best_model)
        self.filer.save_validataion_data(config_space=config_space,
                                         predictions=validation_data,
                                         metric_scores=error_metric,
                                         output_path=output_path,
                                         run_id=self.run_id)

        # calculate mean error_metric
        validation_error = error_metric.mean()

        self.logger.info('Run ID: ' + str(self.run_id) + ' | ' + str(config_space.get('algo')) + ' | ' +
                         self.optimization_metric.__name__ + ': ' + str(validation_error))
        self.logger.debug(str(config_space))

        # return error_metric
        if self.minimize_error_metric_val:
            return validation_error, best_model
        else:
            return 1 - validation_error, best_model
