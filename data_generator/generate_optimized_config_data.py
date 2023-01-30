from lkauto.lkauto import find_best_explicit_configuration
from lenskit.datasets import ML100K
from lenskit.datasets import ML1M
from lenskit.datasets import MovieLens
from lenskit.datasets import ML10M
from lenskit import crossfold as xf
from lenskit.metrics.predict import rmse
from lkauto.utils import filer
import json
import numpy as np
import pandas as pd
import time
import sys

args = sys.argv
dataset = None

if args[1] == 'movielens_small':
    dataset = MovieLens(path='/mnt/data/ml-latest-small').ratings
elif args[1] == 'movielens_1m':
    dataset = ML1M(path='/mnt/data/ml-1m').ratings
elif args[1] == 'movielens_100k':
    dataset = ML100K(path='/mnt/data/ml-100k').ratings
elif args[1] == 'movielens_10m':
    dataset = ML10M(path='/mnt/data/ml-10M100K').ratings
else:
    dataset = pd.read_csv('/mnt/data/preprocessed_data/{}.csv'.format(args[1]))

dataset.name = args[1]
print(dataset.name)

time_limit = int(args[2])
root_mean_square_errors = np.array([])
time_taken = 0
prediction_dataframe = dataset.copy()
fold_predictions = []

for i, tp in enumerate(xf.partition_rows(dataset, 5, rng_spec=42)):
    train_split = tp.train.copy()
    X_test_split = tp.test.copy()
    X_test_split.drop('rating', inplace=True, axis=1)
    y_test_split = tp.test.copy()
    y_test_split = y_test_split[['rating']].iloc[:, 0]

    model, best_config = find_best_explicit_configuration(train_split,
                                                          time_limit_in_sec=time_limit,
                                                          output_dir='/mnt/output/smac_output/{}/{}'.format(dataset.name, i))

    dict = best_config.get_dictionary()

    with open('/mnt/output/configurations/{}_fold{}_config.json'.format(dataset.name, str(i)), 'w') as f:
        json.dump(dict, f)

    st = time.time()
    model.fit(train_split)
    time_taken = time.time() - st

    predictions = model.predict(X_test_split)
    root_mean_square_errors = np.append(root_mean_square_errors,
                                        rmse(predictions, y_test_split, missing='ignore'))
    np.savetxt('/mnt/output/rmse/{}_{}_{}'.format('lenskit-auto', dataset.name, i), root_mean_square_errors)

    # Add predictions to prediction dataframe
    predictions = predictions.to_frame()
    predictions['fold_indicator'] = i
    predictions['index'] = X_test_split.index
    fold_predictions = fold_predictions + predictions.values.tolist()

prediction_dataframe = pd.DataFrame(fold_predictions, columns=['predictions', 'fold', 'index'])
prediction_dataframe['index'] = pd.to_numeric(prediction_dataframe['index'], downcast='integer')
prediction_dataframe['fold'] = pd.to_numeric(prediction_dataframe['fold'], downcast='integer')
prediction_dataframe.index = list(prediction_dataframe['index'])
prediction_dataframe.drop('index', inplace=True, axis=1)

filer.save_predicions_to_csv_file('/mnt/output/predictions/cash_model_{}_predictions.csv'.format(dataset.name),
                                  dataset,
                                  prediction_dataframe)

datapoint = pd.DataFrame([(dataset.name, 'lenskit-auto', root_mean_square_errors.mean(),
                          'optimized_configuration', time_taken, time.time())],
                         columns=['dataset', 'model', 'rmse', 'category', 'time_in_sec', 'timestamp'])
filer.append_data_to_csv(datapoint, '/mnt/output/auto-lenskit_results.csv')

