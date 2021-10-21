import pandas as pd


def min_ratings_per_user(df: pd.DataFrame, num_ratings: int, count_duplicates: bool = False):
    """Prune users with less than num_ratings ratings

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with columns "user", "item", "rating"
    num_ratings: int
        Minimum number of ratings per user
    count_duplicates: bool = False
        If True, all ratings are counted, otherwise only unique ratings are counted

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "user", "item", "rating"
    """
    # get all relevant user_ids
    uids = (
        df['user']
        if count_duplicates
        else df.drop_duplicates(['user', 'item'])['user']
    )
    cnt_items_per_user = uids.value_counts()
    users_of_interest = list(cnt_items_per_user[cnt_items_per_user >= num_ratings].index)

    return df[df['user'].isin(users_of_interest)]


def max_ratings_per_user(df: pd.DataFrame, num_ratings: int, count_duplicates: bool = False):
    """Prune users with more than num_ratings ratings

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with columns "user", "item", "rating"
    num_ratings: int
        Minimum number of ratings per user
    count_duplicates: bool = False
        If True, all ratings are counted, otherwise only unique ratings are counted

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "user", "item", "rating"
    """
    # get all relevant user_ids
    uids = (
        df['user']
        if count_duplicates
        else df.drop_duplicates(['user', 'item'])['user']
    )
    cnt_items_per_user = uids.value_counts()
    users_of_interest = list(cnt_items_per_user[cnt_items_per_user <= num_ratings].index)

    return df[df['user'].isin(users_of_interest)]
