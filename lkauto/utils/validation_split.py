import pandas as pd
import numpy as np
# from lenskit.crossfold import partition_rows
from lenskit.splitting import crossfold_records

def validation_split(data: pd.DataFrame, strategie: str = 'user_based', num_folds: int = 1,
                     frac: float = 0.25, random_state=42) -> dict:
    """
    Returns a dictionary with the indices of the train and validation split for the given data.
    The dictionary has the following structure:
    {
        0: {    # fold 0
            "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "validation": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        },
        1: {    # fold 1
            "train": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "validation": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    }

    Parameters
    ----------
    data : pd.DataFrame
        Pandas Dataframe with the data to be split.
    strategie : str
        cross validation strategie (user_based or row_based)
    num_folds : int
        number of folds for the validation split cross validation
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    dict
        dictionary with the indices of the train and validation split for the given data.
    """
    # decide which validation split strategie to use
    if strategie == 'user_based':
        return user_based_validation_split(data=data, num_folds=num_folds, frac=frac, random_state=random_state)
    elif strategie == 'row_based':
        return row_based_validation_split(data=data, num_folds=num_folds, frac=frac, random_state=random_state)
    else:
        raise ValueError(f"Unknown validation split strategie: {strategie}")


def row_based_validation_split(data: pd.DataFrame, num_folds: int = 1, frac: float = 0.25, random_state=42) -> dict:
    """
    Returns a dictionary with the indices of the train and validation split for the given data.
    The dictionary has the following structure:
    {
        0: {    # fold 0
            "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "validation": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        },
        1: {    # fold 1
            "train": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "validation": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    }

    Parameters
    ----------
    data : pd.DataFrame
        Pandas Dataframe with the data to be split.
    num_folds : int
        number of folds for the validation split cross validation
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    dict
        dictionary with the indices of the train and validation split for the given data.
    """
    # initialize a dictionary with the indices of the train and validation split for the given data
    fold_indices = {i: {"train": np.array([]), "validation": np.array([])} for i in
                    range(num_folds)}
    # if num_folds < 2, we use a holdout validation split
    if num_folds < 2:
        fold_indices = __holdout_validation_split(fold_indices=fold_indices,
                                                  data=data,
                                                  frac=frac,
                                                  random_state=random_state)
    # if num_folds > 1, we use a cross validation split
    else:
        fold_indices = __row_based_k_fold_validation_split(fold_indices=fold_indices,
                                                           data=data,
                                                           num_folds=num_folds,
                                                           random_state=random_state)
    return fold_indices


def user_based_validation_split(data: pd.DataFrame, num_folds: int = 1, frac: float = 0.25, random_state=42) -> dict:
    """
    Returns a dictionary with the indices of the train and validation split for the given data.
    The dictionary has the following structure:
    {
        0: {    # fold 0
            "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "validation": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        },
        1: {    # fold 1
            "train": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "validation": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    }

    Parameters
    ----------
    data : pd.DataFrame
        Pandas Dataframe with the data to be split.
    num_folds : int
        number of folds for the validation split cross validation
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    dict
        dictionary with the indices of the train and validation split for the given data.
    """
    # initialize a dictionary with the indices of the train and validation split for the given data
    fold_indices = {i: {"train": np.array([]), "validation": np.array([])} for i in
                    range(num_folds)}

    # group by users and then sample from each user
    for user, items in data.groupby("user"):
        # if num_folds < 2, we use a holdout validation split
        if num_folds < 2:
            fold_indices = __holdout_validation_split(fold_indices=fold_indices,
                                                      data=items,
                                                      random_state=random_state,
                                                      frac=frac)
        # if num_folds > 1, we use a cross validation split
        else:
            fold_indices = __user_based_crossfold_validation_split(fold_indices=fold_indices,
                                                                   data=items,
                                                                   num_folds=num_folds)

    return fold_indices


def __holdout_validation_split(fold_indices: dict, data: pd.DataFrame, frac: float, random_state=42):
    """
    Returns a dictionary with the indices of the train and validation split for the given data.

    Parameters
    ----------
    fold_indices : dict
        dictionary with the indices of the train and validation split for the given data.
    data : pd.DataFrame
        Pandas Dataframe with the data to be split.
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    dict
    """
    # sample the validation set
    validation = data.sample(frac=frac, random_state=random_state)
    # get the train set by dropping the validation set
    train = data.drop(validation.index)
    # append the indices of the train and validation set to the dictionary
    fold_indices[0]['train'] = np.append(fold_indices[0]["train"], train.index)
    fold_indices[0]['validation'] = np.append(fold_indices[0]["validation"], validation.index)
    # return the dictionary
    return fold_indices


def __row_based_k_fold_validation_split(fold_indices: dict, data: pd.DataFrame, num_folds: int, random_state):
    """
    Returns a dictionary with the indices of the row based cv train and validation split for the given data.

    Parameters
    ----------
    fold_indices : dict
        dictionary with the indices of the train and validation split for the given data.
    data : pd.DataFrame
        Pandas Dataframe with the data to be split.
    """
    # generate the indices of the train and validation split for the given data
    for i, splits in enumerate(crossfold_records(data, partitions=num_folds, rng_spec=random_state)):
        fold_indices[i]['train'] = np.append(fold_indices[i]["train"], splits[0].index)
        fold_indices[i]['validation'] = np.append(fold_indices[i]["validation"], splits[1].index)
    return fold_indices


def __user_based_crossfold_validation_split(fold_indices, data, num_folds) -> dict:
    """
    Returns a dictionary with the indices of the user based cv train and validation split for the given data.

    Parameters
    ----------
    fold_indices : dict
        dictionary with the indices of the train and validation split for the given data.
    data : pd.DataFrame
        Pandas Dataframe with the data to be split.
    num_folds : int
        number of folds for the validation split cross validation

    Returns
    -------
    dict
    """
    # generate splits of equal size
    splits = np.array_split(data, num_folds)
    # go through each split
    for i in range(len(splits)):
        # the split denoted by i is the test set, so all other splits are the train set
        train = pd.concat(splits[:i] + splits[i + 1:], axis=0, ignore_index=False)
        # the test data is simply the index we are currently observing
        test = splits[i]
        # append the indices to the dictionary
        fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train.index)
        fold_indices[i]["validation"] = np.append(fold_indices[i]["validation"], test.index)

    return fold_indices
