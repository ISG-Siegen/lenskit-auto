from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import InCondition
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.svd import BiasedSVD


def get_implicit_default_configuration_space():
    cs = ConfigurationSpace()
    regressor = CategoricalHyperparameter('regressor', ['ItemItem',
                                                        'FunkSVD',
                                                        'UserUser',
                                                        'ImplicitMF',
                                                        'BiasedSVD'
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

    # FunkSVD
    hyperparameter_list = FunkSVD.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['FunkSVD']))

    # ImplicitMF
    hyperparameter_list = ImplicitMF.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['ImplicitMF']))

    # BiasedSVD
    hyperparameter_list = BiasedSVD.get_default_configspace_hyperparameters()
    cs.add_hyperparameters(hyperparameter_list)
    for hyperparameter in hyperparameter_list:
        cs.add_condition(InCondition(hyperparameter, regressor, values=['BiasedSVD']))

    return cs
