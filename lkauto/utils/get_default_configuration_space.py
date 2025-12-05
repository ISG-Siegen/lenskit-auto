from typing import Iterator, Union
from ConfigSpace import Categorical
from ConfigSpace import ConfigurationSpace
from lenskit.data import Dataset
from lenskit.splitting import TTSplit

from lkauto.algorithms.als import BiasedMF
from lkauto.algorithms.als import ImplicitMF
from lkauto.algorithms.bias import Bias
from lkauto.algorithms.funksvd import FunkSVD
from lkauto.algorithms.item_knn import ItemItem
from lkauto.algorithms.svd import BiasedSVD
from lkauto.algorithms.user_knn import UserUser


def get_default_configuration_space(data: Union[Dataset, Iterator[TTSplit]],
                                    feedback: str,
                                    val_fold_indices=None,
                                    validation: Iterator[TTSplit] = None,
                                    random_state=42) -> ConfigurationSpace:
    """
        returns the default configuration space for all included rating prediction algorithms

        Parameters
        ----------
        data: Union[Dataset, Iterator[TTSplit]]
            data to use (either a Dataset or a Train-Test-Pair)
        val_fold_indices
            validation fold indices
        validation: Iterator[TTSplit]
            validation data (provided by user)
        feedback : str
            feedback type (either 'explicit' or 'implicit')
        random_state: int
            random state to use

        Returns
        ------
        ConfigurationSpace
            The default configuration space
    """

    if feedback == 'explicit':
        algo_list = ['ItemItem', 'UserUser', 'FunkSVD', 'BiasedSVD', 'ALSBiasedMF', 'Bias']
    elif feedback == 'implicit':
        algo_list = ['ItemItem', 'UserUser', 'ImplicitMF']
    else:
        raise ValueError("Unknown feedback type: {}".format(feedback))

    # get minimum number of items and users for the given train split
    # num_items = 0
    # num_users = 0

    # if validation is None and not isinstance(data, Dataset): # what does validation is None has to do with the data type???
    #     # Case 1: data is a Train-Test-Split iterator
    #     for fold in data:
    #         if fold.train.item_count < num_items or num_items == 0:
    #             num_items = fold.train.item_count
    #         if fold.train.user_count < num_users or num_users == 0:
    #             num_users = fold.train.user_count
    # else:
    #     # Case 2: data is a Dataset
    #     if data.item_count < num_items or num_items == 0:
    #         num_items = data.item_count
    #     if data.user_count < num_users or num_users == 0:
    #         num_users = data.user_count


    # get minimum number of items and users from data
    if isinstance(data, Dataset):
        # Case 1: data is a Lenskit Dataset
        num_items = data.item_count
        num_users = data.user_count
    else:
        # Case 2: data is a TTSplit iterator
        # convert iterator to list to use it more than once
        folds_list = list(data)
        # get minimum number of items and users
        num_items = min(fold.train.item_count for fold in folds_list)
        num_users = min(fold.train.user_count for fold in folds_list)


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
