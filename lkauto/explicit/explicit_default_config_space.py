from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import InCondition
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF as ALSBiasedMF
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.svd import BiasedSVD


def get_explicit_default_configuration_space():
    cs = ConfigurationSpace()
    regressor = CategoricalHyperparameter('regressor', ['ItemItem',
                                                        'UserUser',
                                                        'ALSBiasedMF',
                                                        'Bias',
                                                        'FunkSVD',
                                                        'BiasedSVD',
                                                        ])
    cs.add_hyperparameter(regressor)

    # ItemItem KNN
    hyperparameter_list = ItemItem.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['ItemItem']))

    # UserUser KNN
    hyperparameter_list = UserUser.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['UserUser']))

    # ALSBiasedMF
    hyperparameter_list = ALSBiasedMF.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['ALSBiasedMF']))

    # Bias
    hyperparameter_list = Bias.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['Bias']))

    # FunkSVD
    hyperparameter_list = FunkSVD.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['FunkSVD']))

    # BiasedSVD
    hyperparameter_list = BiasedSVD.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['BiasedSVD']))

    return cs
