from preprocessing.data_handler import preprocess_all_datasets

# Run all preprocessing steps for all (in the config file) selected datasets.
# local_execution=True makes this run in the local environment and not in a docker container.
#   For the usage in a docker container, this script would need to be executed in the mgmt container
#   with the correct settings to guarantee read/write access.
#
#   For local usage, the script needs to be executed with an environment that has all requirements of the
#   mgmt_requirements.txt
preprocess_all_datasets('data/', ['yelp',
                                  'netflix',
                                  'food',
                                  'amazon-all-beauty',
                                  'amazon-digital-music',
                                  'amazon-fashion',
                                  'amazon-gift-cards',
                                  'amazon-books',
                                  'amazon-software',
                                  'rekko'])
