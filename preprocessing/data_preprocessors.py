import os
import pandas as pd
from utils.yelp_dataset_utils import get_superset_of_column_names_from_file, read_and_write_file
from utils.netflix_dataset_utils import read_netflix_data


def get_all_preprocess_functions(to_preprocess):
    to_preprocess_list = []
    # Maps names of datasets to preprocess function for dataset
    name_to_function_map = {
        "yelp": preprocess_yelp,
        "netflix": preprocess_netflix,
        "food": preprocess_food
    }

    # Add amazon functions (which are dynamic objects)
    for amazon_function in build_amazon_load_functions():
        name_to_function_map[amazon_function.__name__[11:]] = amazon_function

    # Filter which preprocess functions to use
    for dataset_name in name_to_function_map:
        if dataset_name in to_preprocess:
            to_preprocess_list.append(name_to_function_map[dataset_name])

    return to_preprocess_list


# ---- Utils
def convert_date_to_timestamp(data, to_encode_columns, prefix=False):
    for col in to_encode_columns:
        df_dates = pd.to_datetime(data[col]).apply(lambda x: int(pd.Timestamp(x).value / 10 ** 9))

        if prefix:
            df_dates.name = "ts_" + col

        data = data.drop([col], axis=1)
        data = pd.concat([data, df_dates], axis=1)
    return data


# -- Yelp
def preprocess_yelp(base_path):
    # We assume the downloaded yelp datasets was extracted into a folder called "yelp"
    filename = 'yelp_academic_dataset_review'

    column_names = get_superset_of_column_names_from_file(os.path.join(base_path, ('yelp/' + filename + '.json')))
    read_and_write_file(os.path.join(base_path, ('yelp/' + filename + '.json')),
                        os.path.join(base_path, ('yelp/' + filename + '.csv')), column_names)

    review_data = pd.read_csv(os.path.join(base_path, 'yelp/yelp_academic_dataset_review.csv'))

    review_data = review_data.drop(['text', 'review_id', 'useful', 'funny', 'cool', 'date'], axis=1)

    data = review_data.rename(columns={'user_id': 'user',
                                       'business_id': 'item',
                                       'stars': 'rating'})

    data['user'] = data.groupby(['user']).ngroup()
    data['item'] = data.groupby(['item']).ngroup()

    return 'yelp', data


# -- Netflix
def preprocess_netflix(base_path):
    # This preprocessing script assume that the downloaded archive folder was re-named to "netflix"
    filenames = ['combined_data_1',
                 'combined_data_2',
                 'combined_data_3',
                 'combined_data_4']

    read_netflix_data(filenames, base_path)

    data = pd.read_csv(os.path.join(base_path, 'netflix/fullcombined_data.csv'))
    data.columns = ['item', 'user', 'rating', 'timestamp']

    data = data.drop(['timestamp'], axis=1)

    return 'netflix', data


# -- Food.com Recipe & Review Data
def preprocess_food(base_path):
    # This preprocessing script assume that the downloaded archive folder was re-named to "food_com_archive"
    interactions_df = pd.read_csv(os.path.join(base_path, "food_com_archive", "RAW_interactions.csv"))

    # -- Preprocess Interactions DF
    interactions_df.drop(columns=["review", "date"], inplace=True)

    data = interactions_df.rename(columns={'user_id': 'user',
                                           'recipe_id': 'item',
                                           'rating': 'rating'})

    return 'foodCom', data


# -- Amazon
def create_amazon_load_function(file_name, dataset_name):
    def amazon_load_function_template(base_path):
        data = pd.read_csv(os.path.join(base_path, 'amazon', '{}.csv'.format(file_name)),
                           names=['item', 'user', 'rating', 'timestamp'])

        data = data.drop(['timestamp'], axis=1)

        data['user'] = data.groupby(['user']).ngroup()
        data['item'] = data.groupby(['item']).ngroup()

        return dataset_name, data

    amazon_load_function_template.__name__ = "preprocess_{}".format(dataset_name)
    return amazon_load_function_template


def build_amazon_load_functions():
    # This preprocessing script assume that the downloaded amazon datasets were moved to a folder was named "amazon"

    # List of Amazon Dataset Meta-info needed to build loader
    amazon_dataset_info = [
        ('All_Beauty', 'amazon-all-beauty'),
        ('Digital_Music', 'amazon-digital-music'),
        ('AMAZON_FASHION', 'amazon-fashion'),
        ('Gift_Cards', 'amazon-gift-cards'),
        ('Books', 'amazon-books'),
        ('Software', 'amazon-software')
    ]

    # For saving function
    load_functions_list = []

    # Build function for each combination and append to list
    for file_name, dataset_name in amazon_dataset_info:
        # Build load function
        load_functions_list.append(create_amazon_load_function(file_name, dataset_name))

    return load_functions_list


