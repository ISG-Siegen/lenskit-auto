#import pandas as pd
from lenskit.data import Dataset,from_interactions_df

def min_ratings_per_user(dataset: Dataset, num_ratings: int, count_duplicates: bool = False):
    """Prune users with less than num_ratings ratings

    Parameters
    ----------
    dataset: Dataset
        LensKit Dataset object containing user-item interactions with ratings
    num_ratings: int
        Minimum number of ratings per user
    count_duplicates: bool = False
        If True, all ratings are counted, otherwise only unique ratings are counted

    Returns
    -------
    Dataset
        Filtered Dataset with only users meeting the minimum rating threshold
        the Dataset will contain the columns "user_id", "item_id", "rating"
    """
    # get the user statistics from the dataset
    user_stats = dataset.user_stats()
    if count_duplicates: 
        valid_users = user_stats[user_stats['count'] >= num_ratings].index # count: total number of ratings (including duplicates)
    else:
        valid_users = user_stats[user_stats['item_count'] >= num_ratings].index # item_count: number of unique items rated
    # convert the interaction table to a pandas DataFrame and filter by valid users
    users_of_interest = dataset.interaction_table(format='pandas', original_ids=True)
    users_of_interest = users_of_interest[users_of_interest['user_id'].isin(valid_users)]
    return from_interactions_df(users_of_interest)




def max_ratings_per_user(dataset: Dataset, num_ratings: int, count_duplicates: bool = False):
    """Prune users with more than num_ratings ratings

    Parameters
    ----------
    dataset: Dataset
        LensKit Dataset object containing user-item interactions with ratings
    num_ratings: int
        Maximum number of ratings per user
    count_duplicates: bool = False
        If True, all ratings are counted, otherwise only unique ratings are counted

    Returns
    -------
    Dataset
        Filtered Dataset with only users meeting the minimum rating threshold
        the Dataset will contain the columns "user_id", "item_id", "rating"
    """

    user_stats = dataset.user_stats()
    if count_duplicates: 
        valid_users = user_stats[user_stats['count'] <= num_ratings].index # count: total number of ratings (including duplicates)
    else:
        valid_users = user_stats[user_stats['item_count'] <= num_ratings].index # item_count: number of unique items rated
    # convert the interaction table to a pandas DataFrame and filter by valid users
    users_of_interest = dataset.interaction_table(format='pandas', original_ids=True)
    users_of_interest = users_of_interest[users_of_interest['user_id'].isin(valid_users)]
    return from_interactions_df(users_of_interest)
