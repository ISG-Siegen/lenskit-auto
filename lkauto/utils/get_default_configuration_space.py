from ConfigSpace import ConfigurationSpace
from ConfigSpace import Categorical
from ConfigSpace.conditions import InCondition
from lkauto.algorithms.user_knn import UserUser
from lkauto.algorithms.item_knn import ItemItem
from lkauto.algorithms.als import BiasedMF as ALSBiasedMF
from lkauto.algorithms.funksvd import FunkSVD
from lkauto.algorithms.bias import Bias
from lkauto.algorithms.svd import BiasedSVD


def get_default_configuration_space(random_state=42) -> ConfigurationSpace:
    """
        returns the default configuration space for all included rating predictions algorithms
    """

    algo_list = ['ItemItem', 'UserUser', 'FunkSVD', 'BiasedSVD', 'ALSBiasedMF', 'Bias']

    # define configuration space
    cs = ConfigurationSpace(
        seed=random_state,
        space={
            "algo": Categorical("algo", algo_list, default="ItemItem"),
        }
    )

    for algo in algo_list:
        if algo == 'UserUser':
            model = UserUser
        elif algo == 'ItemItem':
            model = ItemItem
        elif algo == 'FunkSVD':
            model = FunkSVD
        elif algo == 'BiasedSVD':
            model = BiasedSVD
        elif algo == 'ALSBiasedMF':
            model = ALSBiasedMF
        elif algo == 'Bias':
            model = Bias
        else:
            raise ValueError("Unknown algorithm: {}".format(algo))

        cs.add_configuration_space(
            prefix=algo,
            delimiter=":",
            configuration_space=model.get_default_configspace(),
            parent_hyperparameter={"parent": cs["algo"], "value": algo},
        )

    return cs
