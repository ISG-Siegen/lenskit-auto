from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import InCondition
from lkpy.lenskit.algorithms.user_knn import UserUser
from lkpy.lenskit.algorithms.item_knn import ItemItem
from lkpy.lenskit.algorithms.als import BiasedMF as ALSBiasedMF
from lkpy.lenskit.algorithms.funksvd import FunkSVD
from lkpy.lenskit.algorithms.bias import Bias
from lkpy.lenskit.algorithms.svd import BiasedSVD


def get_explicit_default_configuration_space() -> ConfigurationSpace:
    """
        returns the default configuration space for all included rating predictions algorithms
    """
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
