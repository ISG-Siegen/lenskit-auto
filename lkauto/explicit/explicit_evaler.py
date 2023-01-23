from ConfigSpace import ConfigurationSpace
from lkauto.utils.filer import Filer
import pandas as pd
from lenskit.metrics.predict import rmse
from lkauto.utils.get_model_from_cs import get_model_from_cs
import numpy as np


class ExplicitEvaler:
    """ExplicitEvaler

        the ExplicitEvaler class handles the evaluation of the optimization tool.
        An Evaluation run consists of training a model and predict the performance on a validation split.

        Attributes
        ----------
        train : pd.DataFrame
            pandas dataset containing the train split.
        filer : Filer
            filer to organize the output.
        random_state :
            The random number generator or seed (see :py:func:`lenskit.util.rng`).
        folds :
            The number of folds of the validation split

        Methods
        ----------
        evaluate_explicit(config_space: ConfigurationSpace) -> float
    """

    def __init__(self, train: pd.DataFrame, filer: Filer, random_state=42, folds: int = None) -> None:
        self.train = train
        self.filer = filer
        self.random_state = random_state
        self.folds = folds
        self.run_id = 0
        self.top_50_runs = pd.DataFrame(columns=['run_id', 'model', 'error'])

    def evaluate_explicit(self, config_space: ConfigurationSpace) -> float:
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
        output_path = 'smac_runs/'
        self.run_id += 1
        error_metric = np.array([])
        validation_data = pd.DataFrame()

        # get model from configuration space
        model = get_model_from_cs(config_space, feedback='explicit')

        # holdout split using pandas and numpy random seed
        validation_train = self.train.sample(frac=0.75, random_state=self.random_state)  # random state is a seed value
        test = self.train.drop(validation_train.index)
        X_validation_test = test.copy()
        y_validation_test = test.copy()

        # process validation split
        X_validation_test = X_validation_test.drop('rating', inplace=False, axis=1)
        y_validation_test = y_validation_test[['rating']].iloc[:, 0]

        # fit and predict model from configuration
        model.fit(validation_train)
        predictions = model.predict(X_validation_test)
        predictions.index = X_validation_test.index

        # calculate error_metric and append to numpy array
        error_metric = np.append(error_metric, rmse(predictions, y_validation_test, missing='ignore'))

        validation_data = pd.concat([validation_data, predictions], axis=0)

        # Save validation data for reproducibility
        self.filer.save_validataion_data(config_space=config_space,
                                         predictions=validation_data,
                                         metric_scores=error_metric,
                                         output_path=output_path,
                                         run_id=self.run_id)

        validation_error = error_metric.mean()

        return validation_error
