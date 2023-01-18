import pandas as pd

from lkauto.lkauto import find_best_explicit_configuration
from lkauto.lkauto import find_best_implicit_configuration
from lenskit.datasets import ML1M
from lenskit.crossfold import sample_rows
from lenskit.metrics.predict import rmse
import lenskit.crossfold as xf
from lkauto.utils.filer import Filer
from lenskit import topn, batch
import os

from numpy import dot
from numpy.linalg import norm
import numpy

from lkauto.utils.meta_information import get_similarities
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter

if __name__ == '__main__':
    # numpy.seterr(all='raise')

    # Load MovieLens Latest Small dataset
    mlsmall = pd.read_csv('data/preprocessed_data/epinions.csv')
    mlsmall.name = 'ml_small'

    """
           ***** Rating Prediction Task Example *****
    """
    # Row based data split
    train, test = sample_rows(mlsmall, None, 25000)

    # # gather data similarity information
    # data_similarities = get_similarities(train, test)

    x_test = test.copy()
    x_test.drop('rating', inplace=True, axis=1)
    y_test = test[['rating']].iloc[:, 0]

    # LensKit-Auto function call
    model, config = find_best_explicit_configuration(train)

    model.fit(train)
    predictions = model.predict(x_test)
    root_mean_square_error = rmse(predictions, y_test)
    print('outer_rmse: {}'.format(root_mean_square_error))
    # print('inner_rmse: {}'.format(validation_score))
    # print('validation_loss: {}'.format(validation_score - root_mean_square_error))
    # print('data_similarities: {}'.format(data_similarities))
    # print('validation_similarities: {}'.format(validation_similarities))
    # print('cosine_similarity: {}'.format(dot(data_similarities, validation_similarities) / (norm(data_similarities) * norm(validation_similarities))))
    # print('inner_rmse')
    # print(config)

    # """
    #        ***** Top-N Ranking Prediction Task Example *****
    # """
    # # User based data-split
    # for i, tp in enumerate(xf.partition_users(mlsmall, 1, xf.SampleN(5))):
    #     train = tp.train.copy()
    #     test = tp.test.copy()
    #
    #     # LensKit-Auto function call
    #     model, config = find_best_implicit_configuration(train)
    #
    #     # fit
    #     model.fit(train)
    #     # recommend
    #     recs = batch.recommend(algo=model, users=test['user'].unique(), n=5)
    #
    #     # initialize RecListAnalysis
    #     rla = topn.RecListAnalysis()
    #     # add precision metric
    #     rla.add_metric(topn.ndcg)
    #
    #     # compute scores
    #     scores = rla.compute(recs, test, include_missing=True)


# def define_custom_config_space():
#     # initialize ConfigurationSpace
#     cs = ConfigurationSpace()
#     # define ItemItem and UserUser as set of algorithms
#     regressor = CategoricalHyperparameter('regressor', ['ItemItem', 'UserUser'])
#     # add the ItemItem and UserUser set to the ConfigurationSpace cs
#     cs.add_hyperparameter(regressor)
#
#     # Get a list of all ItemItem parameters of the provided ItemItem.get_default_configspace_hyperparameters() function
#     hyperparameter_list = ItemItem.get_default_configspace_hyperparameters()
#     # add all hyperparameters of hyperparameterlist to the ConfigurationSpace
#     cs.add_hyperparameters(hyperparameter_list)
#     # Set the condition that all ItemItem hyperparameters are bound to the ItemItem algorithm
#     for hyperparameter in hyperparameter_list:
#         cs.add_condition(InCondition(hyperparameter, regressor, values=['ItemItem']))
#
#     # Other than the ItemItem case, we can define the default_configspace_parameters by hand.
#     user_user_min_nbrs = UniformIntegerHyperparameter('user_user_min_nbrs', lower=1, upper=50, default_value=1)
#     user_user_min_sim = UniformFloatHyperparameter('user_user_min_sim', lower=0, upper=0.1, default_value=0)
#
#     # add the UserUser hyperparameters to the ConfigSpace cs
#     cs.add_hyperparameters([user_user_min_sim, user_user_min_nbrs])
#
#     # Set the condition that all UserUser hyperparameters are bound to the UserUser algorithm
#     cs.add_condition(InCondition(user_user_min_sim, regressor, values=['UserUser']))
#     cs.add_condition(InCondition(user_user_min_nbrs, regressor, values=['UserUser']))

