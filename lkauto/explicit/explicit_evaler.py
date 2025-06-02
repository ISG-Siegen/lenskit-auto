import logging

import numpy as np
import pandas as pd

from typing import Iterator
from lenskit.data import Dataset
from lenskit.pipeline import predict_pipeline, topn_pipeline
from lenskit.batch import recommend
from lenskit.metrics import RunAnalysis
from lenskit.splitting import TTSplit
from ConfigSpace import ConfigurationSpace
from sklearn.model_selection import train_test_split

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
        train : pd.DataFrame
            pandas dataset containing the train split.
        optimization_metric: function
            LensKit prediction accuracy metric used to evaluate the model (either rmse or mae)
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
        ensemble_size :
            The number of models used to build the final ensemble predictor.
        minimize_error_metric_val :
            If True, the error metric is minimized. If False, the error metric is maximized. This parameter needs to be
            set in corelation with the optimization metric.

        Methods
        ----------
        evaluate_explicit(config_space: ConfigurationSpace) -> float
    """

    def __init__(self,
                 data: Dataset,
                 train: pd.DataFrame,
                 optimization_metric,
                 filer: Filer,
                 ttsplits: Iterator[TTSplit] = None,
                 validation=None,
                 random_state=42,
                 split_folds: int = 1,
                 split_strategie: str = 'user_based',
                 split_frac: float = 0.25,
                 ensemble_size: int = 50,
                 minimize_error_metric_val: bool = True,
                 ) -> None:
        self.logger = logging.getLogger('lenskit-auto')
        self.data = data
        self.train = train
        self.filer = filer
        self.ttsplits = ttsplits
        self.validation = validation
        self.random_state = random_state
        self.split_folds = split_folds
        self.optimization_metric = optimization_metric
        self.split_strategie = split_strategie
        self.split_frac = split_frac
        self.minimize_error_metric_val = minimize_error_metric_val
        self.run_id = 0
        self.ensemble_size = ensemble_size
        self.top_n_runs = pd.DataFrame(columns=['run_id', 'model', 'error'])
        if self.ttsplits is None:
            self.train_test_splits = validation_split(data=self.data,
                                                      strategy=self.split_strategie,
                                                      num_folds=self.split_folds,
                                                      frac=self.split_frac,
                                                      random_state=self.random_state)
        else:
            self.train_test_splits = self.ttsplits

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
        error_metric = np.array([])
        validation_data = pd.DataFrame()

        # get model from configuration space
        model = get_model_from_cs(config_space, feedback='explicit')

        '''
        # loop over validation folds
        for fold in range(self.split_folds):
            if self.validation is None:
                # get validation split by fold index
                validation_train = self.train.loc[self.train_test_splits[fold]["train"], :]
                validation_test = self.train.loc[self.train_test_splits[fold]["validation"], :]
            else:
                validation_train = self.train
                validation_test = self.validation

            # split validation data into X and y
            x_validation_test = validation_test.copy()
            y_validation_test = validation_test.copy()

            # process validation split
            x_validation_test = x_validation_test.drop('rating', inplace=False, axis=1)
            y_validation_test = y_validation_test[['rating']].iloc[:, 0]


            # fit and predict model from configuration
            model.fit(validation_train)
            predictions = model.predict(x_validation_test)
            predictions.index = x_validation_test.index

            # calculate error_metric and append to numpy array
            error_metric = np.append(error_metric,
                                     self.optimization_metric(predictions, y_validation_test, missing='ignore'))

            validation_data = pd.concat([validation_data, predictions], axis=0)
            '''

        for fold in self.train_test_splits:
            validation_train = fold.train
            validation_test = fold.test

            pipeline = predict_pipeline(scorer=model)
            fit_pipeline = pipeline.clone()
            fit_pipeline.train(data=validation_train)

            recs = recommend(fit_pipeline, validation_test.keys())

            run_analysis = RunAnalysis()
            run_analysis.add_metric(self.optimization_metric)
            error_results = run_analysis.measure(recs, validation_test)

            error_metric = np.append(error_metric, error_results)
            validation_data = pd.concat([validation_data, recs], ignore_index=True)

        # Save validation data for reproducibility and ensembling
        self.top_n_runs = update_top_n_runs(config_space=config_space,
                                            errors=error_metric,
                                            num_models=self.ensemble_size,
                                            run_id=self.run_id,
                                            top_n_runs=self.top_n_runs)
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
            return validation_error
        else:
            return 1 - validation_error
