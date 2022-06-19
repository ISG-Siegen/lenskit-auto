import os
import json
import glob
import pandas as pd
from preprocessing.data_preprocessors import get_all_preprocess_functions


# --- Utils
def save_to_files(base_path, name, data_df):
    print("######## Store Datasets {} to CSV ########".format(name))

    # drop None values
    pre_drop_len = len(data_df)
    data_df = data_df.dropna()
    print("Dropped {} None".format(pre_drop_len - len(data_df)))

    # drop duplicates
    pre_drop_len = len(data_df)
    data_df = data_df.drop_duplicates(ignore_index=True)
    print("Dropped {} duplicates".format(pre_drop_len - len(data_df)))

    # drop multiple ratings of a user on a single item
    pre_drop_len = len(data_df)
    data_df = data_df.drop_duplicates(subset=['user', 'item'], keep='first')
    print("Dropped {} duplicate ratings".format(pre_drop_len - len(data_df)))

    # print dataset_length
    print("Number of Reviews {}".format(len(data_df)))

    # Rename columns to standardized format
    file_path_csv = os.path.join(base_path, "preprocessed_data/{}.csv".format(name))

    # Add meta data as empty columns
    data_df.to_csv(path_or_buf=file_path_csv, sep=',', header=True, index=False)


def preprocess_all_datasets(path, to_preprocess):
    preprocessors = get_all_preprocess_functions(to_preprocess)

    n_preprocessors = len(preprocessors)

    for idx, fn in enumerate(preprocessors, 1):
        print("Start Preprocessing: {} [{}/{}]".format(fn.__name__, idx, n_preprocessors))
        # Preprocess and save results to csv
        save_to_files(path, *fn(path))
