from ConfigSpace import ConfigurationSpace
from ConfigSpace import Categorical
from lkauto.algorithms.user_knn import UserUser
from lkauto.algorithms.item_knn import ItemItem
from lkauto.algorithms.als import BiasedMF as ALSBiasedMF
from lkauto.algorithms.funksvd import FunkSVD
from lkauto.algorithms.bias import Bias
from lkauto.algorithms.svd import BiasedSVD
from lkauto.algorithms.als import ImplicitMF


def get_default_configuration_space(feedback: str, random_state=42) -> ConfigurationSpace:
    """
        returns the default configuration space for all included rating predictions algorithms

        Parameters
        ----------
        feedback : str
            feedback type, either 'explicit' or 'implicit'
        random_state: int
            random state to use
    """

    if feedback == 'explicit':
        algo_list = ['ItemItem', 'UserUser', 'FunkSVD', 'BiasedSVD', 'ALSBiasedMF', 'Bias']
    elif feedback == 'implicit':
        algo_list = ['ItemItem', 'FunkSVD', 'UserUser', 'ImplicitMF', 'BiasedSVD']
    else:
        raise ValueError("Unknown feedback type: {}".format(feedback))

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
        elif algo == 'ImplicitMF':
            model = ImplicitMF
        else:
            raise ValueError("Unknown algorithm: {}".format(algo))

        cs.add_configuration_space(
            prefix=algo,
            delimiter=":",
            configuration_space=model.get_default_configspace(),
            parent_hyperparameter={"parent": cs["algo"], "value": algo},
        )

    return cs
