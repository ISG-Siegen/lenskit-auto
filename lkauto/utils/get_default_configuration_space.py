import pandas as pd
from ConfigSpace import Categorical
from ConfigSpace import ConfigurationSpace

from lkauto.algorithms.als import BiasedMF
from lkauto.algorithms.als import ImplicitMF
from lkauto.algorithms.bias import Bias
from lkauto.algorithms.funksvd import FunkSVD
from lkauto.algorithms.item_knn import ItemItem
from lkauto.algorithms.svd import BiasedSVD
from lkauto.algorithms.user_knn import UserUser


def get_default_configuration_space(data: pd.DataFrame,
                                    val_fold_indices,
                                    feedback: str,
                                    validation: pd.DataFrame = None,
                                    random_state=42) -> ConfigurationSpace:
    """
        returns the default configuration space for all included rating predictions algorithms

        Parameters
        ----------
        data: pd.DataFrame
            data to use
        val_fold_indices
            validation fold indices
        validation: pd.DataFrame
            validation data (provided by user)
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

    # get minimum number of items and users for the given train split

    num_items = 0
    num_users = 0
    if validation is None:
        val_fold_indices = val_fold_indices
        for fold in range(len(val_fold_indices)):
            tmp = data.loc[val_fold_indices[fold]["train"], :]
            if tmp['item'].nunique() < num_items or num_items == 0:
                num_items = tmp['item'].nunique()
            if tmp['user'].nunique() < num_users or num_users == 0:
                num_users = tmp['user'].nunique()
    else:
        if data['item'].nunique() < num_items or num_items == 0:
            num_items = data['item'].nunique()
        if data['user'].nunique() < num_users or num_users == 0:
            num_users = data['user'].nunique()

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
            model = BiasedMF
        elif algo == 'Bias':
            model = Bias
        elif algo == 'ImplicitMF':
            model = ImplicitMF
        else:
            raise ValueError("Unknown algorithm: {}".format(algo))

        # add configuration space of algorithm
        cs.add_configuration_space(
            prefix=algo,
            delimiter=":",
            configuration_space=model.get_default_configspace(number_user=num_users, number_item=num_items),
            parent_hyperparameter={"parent": cs["algo"], "value": algo},
        )

    return cs
