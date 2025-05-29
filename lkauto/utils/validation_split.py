from typing import Iterator

import pandas as pd
from lenskit.splitting import crossfold_records, crossfold_users, sample_records, SampleFrac, TTSplit
from lenskit.data import Dataset


def validation_split(data: Dataset, strategy: str = 'user_based', num_folds: int = 1,
                     frac: float = 0.25, random_state=42) -> Iterator[TTSplit]:
    """
    Returns the Train-Test-Split for the given Dataset

    Parameters
    ----------
    data : Dataset
        Lenskit Dataset with the data to be split.
    strategy : str
        cross validation strategy (user_based or row_based)
    num_folds : int
        number of folds for the validation split cross validation
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    Iterator[TTSplit]
        The Train-Test-Split for the given Dataset
    """
    # decide which validation split strategy to use
    if strategy == 'user_based':
        return user_based_validation_split(data=data, num_folds=num_folds, frac=frac, random_state=random_state)
    elif strategy == 'row_based':
        return row_based_validation_split(data=data, num_folds=num_folds, frac=frac, random_state=random_state)
    else:
        raise ValueError(f"Unknown validation split strategy: {strategy}")


def row_based_validation_split(data: Dataset, num_folds: int = 1, frac: float = 0.25, random_state=42) -> Iterator[TTSplit]:
    """
    Returns a Train-Test-Split for the given data.

    Parameters
    ----------
    data : Dataset
        Lenskit Dataset with the data to be split.
    num_folds : int
        number of folds for the validation split cross validation
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    Iterator[TTSPlit]
        Train-Tet-Split for the given data.
    """

    if num_folds < 2:
        return __holdout_validation_split(data=data, frac=frac, random_state=random_state)
    else:
        return __row_based_k_fold_validation_split(data=data, num_folds=num_folds, random_state=random_state)


def user_based_validation_split(data: Dataset, num_folds: int = 1, frac: float = 0.25, random_state=42) -> Iterator[
    TTSplit]:
    """
    Parameters
    ----------
    data : Dataset
        Lenskit Dataset with the data to be split.
    num_folds : int
        number of folds for the validation split cross validation
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    Iterator[TTSplit]
        Train-Test-Split for the given data.
    """

    if num_folds < 2:
        return __holdout_validation_split(data=data, frac=frac, random_state=random_state)
    else:
        return __user_based_crossfold_validation_split(data=data, num_folds=num_folds)



def __holdout_validation_split(data: Dataset, frac: float, random_state=42):
    """
    Returns a Train-Test-Split for the given data.

    Parameters
    ----------
    data : Dataset
        Lenskit Dataset with the data to be split.
    frac : float
        fraction of the dataset to be used for the validation split. If num_folds > 1, the fraction value
        will be ignored.
    random_state : int
        random state for the validation split

    Returns
    -------
    Iterator[TTSplit]
        Train-Test-Split for the given data. Should only contain one fold.
    """

    splits = sample_records(data=data, size=int(data.interaction_count * frac), rng=random_state)

    if hasattr(splits, "_iter__"):
        return splits
    else:
        return iter([splits])


def __row_based_k_fold_validation_split(data: Dataset, num_folds: int, random_state):
    """
    Returns a Train-Test-Split for the given data.

    Parameters
    ----------
    data : Dataset
        Lenskit Dataset with the data to be split.
    """

    splits = crossfold_records(data=data, partitions=num_folds, rng=random_state)
    return splits



def __user_based_crossfold_validation_split(data: Dataset, num_folds) -> Iterator[TTSplit]:
    """
    Returns a Train-Test-Split for the given data.

    Parameters
    ----------
    data : Dataset
        Pandas Dataframe with the data to be split.
    num_folds : int
        number of folds for the validation split cross validation

    Returns
    -------
    Iterator[TTSplit]
        Train-Test-Split for the given data.
    """

    return crossfold_users(data=data, partitions=num_folds, method=SampleFrac(0.2))
