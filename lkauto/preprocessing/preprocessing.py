import logging

from lkauto.preprocessing.pruning import min_ratings_per_user, max_ratings_per_user
from lenskit.data import Dataset
from lenskit.data import from_interactions_df


def preprocess_data(data: Dataset,
                    user_col: str,
                    item_col: str,
                    rating_col: str = None,
                    timestamp_col: str = None,
                    include_timestamp: bool = True,
                    drop_na_values: bool = True,
                    drop_duplicates: bool = True,
                    min_interactions_per_user: int = None,
                    max_interactions_per_user: int = None) -> Dataset:
    """Preprocess data for LensKit
    This method can perform the following steps based on the user input:
    1. rename columns to "user", "item", "rating", "timestamp"
    2. Drop all rows with NaN values
    3. Drop all duplicate rows
    4. Drop all users with less than min_interactions_per_user interactions
    5. Drop all users with more than max_interactions_per_user interactions

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with columns "user", "item", "rating"
    user_col: str
        Name of the user column
    item_col: str
        Name of the item column
    rating_col: str
        Name of the rating column
    timestamp_col: str
        Name of the timestamp column
    include_timestamp: bool = True
        If True, the timestamp column will be included in the dataset
    drop_na_values: bool = True
        If True, all rows with NaN values will be dropped
    drop_duplicates: bool = True
        If True, all duplicate rows will be dropped
    min_interactions_per_user: int = None
        If not None, all users with less than this number of interactions will be dropped
    max_interactions_per_user: int = None
        If not None, all users with more than this number of interactions will be dropped

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "user", "item", "rating"
    """

    logger = logging.getLogger('lenskit-auto')
    logger.info('--Start Preprocessing--')

    data = data.interaction_table(format='pandas')
    # original_cols = data.columns.tolist()
    # print(original_cols)

    # rename columns
    if include_timestamp:
        if rating_col is None:
            data = data[[user_col, item_col, timestamp_col]]
            data.columns = ['user', 'item', 'timestamp']
        else:
            # data = data[[user_col, item_col, rating_col, timestamp_col]]
            data.columns = ['user', 'item', 'rating', 'timestamp']
    else:
        if rating_col is None:
            data = data[[user_col, item_col]]
            data.columns = ['user', 'item']
        else:
            data = data[[user_col, item_col, rating_col]]
            data.columns = ['user', 'item', 'rating']

    # drop rows with NaN values
    if drop_na_values:
        logger.debug('Dropping rows with NaN values...')
        data = data.dropna()

    # drop duplicate rows
    if drop_duplicates:
        logger.debug('Dropping duplicate rows...')
        data = data.drop_duplicates(keep='first', inplace=False)

    # drop users with less than min_interactions_per_user interactions
    if min_interactions_per_user is not None:
        logger.debug('Dropping users with less than {} interactions...'.format(min_interactions_per_user))
        data = min_ratings_per_user(data, min_interactions_per_user)

    # drop users with more than max_interactions_per_user interactions
    if max_interactions_per_user is not None:
        logger.debug('Dropping users with more than {} interactions...'.format(max_interactions_per_user))
        data = max_ratings_per_user(data, max_interactions_per_user)

    logger.info('--End Preprocessing--')

    return from_interactions_df(data)
