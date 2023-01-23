from lkauto.utils.get_model_from_cs import get_model_from_cs
import lenskit.crossfold as xf
from ConfigSpace import ConfigurationSpace
from lenskit import topn, batch
from lkauto.utils.filer import Filer
import numpy as np
import pandas as pd


class ImplicitEvaler:
    """ImplicitEvaler

            the ImplicitEvaler class handles the evaluation of the optimization tool.
            An Evaluation run consists of training a model and predict the performance on a validation split.

            Attributes
            ----------
            train : pd.DataFrame
                pandas dataset containing the train split.
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
    def __init__(self, train: pd.DataFrame, filer: Filer, random_state=42, folds: int = 1) -> None:
        self.train = train
        self.random_seed = random_state
        self.folds = folds
        self.filer = filer
        self.run_id = 0

    def evaluate_implicit(self, config_space: ConfigurationSpace) -> float:
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
        precisions = np.array([])
        validation_data = pd.DataFrame()

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
            rla.add_metric(topn.precision)

            # compute scores
            scores = rla.compute(recs, validation_test, include_missing=True)

            # store data
            validation_data = pd.concat([validation_data, recs], axis=0)
            precisions = np.append(precisions, scores['precision'].mean())

        # save validation data
        self.filer.save_validataion_data(config_space=config_space,
                                         predictions=validation_data,
                                         metric_scores=precisions,
                                         output_path=output_path,
                                         run_id=self.run_id)

        # FIXME: is this correct? I guess the evaluation takes just the last fold into account
        # store score mean and subtract by 1 to enable SMAC to minimize returned value
        validation_error = 1 - scores['precision'].mean()

        return validation_error

