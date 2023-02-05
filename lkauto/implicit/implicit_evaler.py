import lenskit.crossfold as xf
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from lenskit import topn, batch

from lkauto.utils.filer import Filer
from lkauto.utils.get_model_from_cs import get_model_from_cs


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
            random_state :
                The random number generator or seed (see :py:func:`lenskit.util.rng`).
            folds : int
                The number of folds of the validation split

            Methods
            ----------
            evaluate_explicit(config_space: ConfigurationSpace) -> float
        """

    def __init__(self, train: pd.DataFrame, optimization_metric, filer: Filer, random_state=42, folds: int = 1) -> None:
        self.train = train
        self.optimization_metric = optimization_metric
        self.random_seed = random_state
        self.folds = folds
        self.filer = filer
        self.run_id = 0

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
        output_path = 'smac_runs/'
        self.run_id += 1
        metric_scores = np.array([])
        validation_data = pd.DataFrame()
        scores = None

        # get model form configuration space
        model = get_model_from_cs(config_space, feedback='implicit')

        # validation split based on users
        for i, tp in enumerate(xf.partition_users(self.train, self.folds, xf.SampleN(5))):
            validation_train = tp.train.copy()
            validation_test = tp.test.copy()

            # fit and recommend from configuration
            model = model.fit(validation_train)
            recs = batch.recommend(algo=model, users=validation_test['user'].unique(), n=5)

            rla = topn.RecListAnalysis()
            rla.add_metric(self.optimization_metric)

            # compute scores
            scores = rla.compute(recs, validation_test, include_missing=True)

            # store data
            validation_data = pd.concat([validation_data, recs], axis=0)
            metric_scores = np.append(metric_scores, scores[self.optimization_metric.__name__].mean())

        # save validation data
        self.filer.save_validataion_data(config_space=config_space,
                                         predictions=validation_data,
                                         metric_scores=metric_scores,
                                         output_path=output_path,
                                         run_id=self.run_id)

        # FIXME: is this correct? I guess the evaluation takes just the last fold into account
        # store score mean and subtract by 1 to enable SMAC to minimize returned value
        validation_error = 1 - scores[self.optimization_metric.__name__].mean()

        return validation_error
