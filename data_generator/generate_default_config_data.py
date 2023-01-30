from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF as ALSBiasedMF
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.basic import Fallback
from lenskit.datasets import ML100K
from lenskit.datasets import ML1M
from lenskit.datasets import MovieLens
from lenskit.datasets import ML10M
from lenskit import crossfold as xf
from lenskit.metrics.predict import rmse
from lkauto.utils import filer
import numpy as np
import pandas as pd
import sys
import time

args = sys.argv
dataset = None
path_prefix = ''

if args[1] == 'movielens_small':
    dataset = MovieLens(path='{}data/ml-latest-small'.format(path_prefix)).ratings
elif args[1] == 'movielens_1m':
    dataset = ML1M(path='{}data/ml-1m'.format(path_prefix)).ratings
elif args[1] == 'movielens_100k':
    dataset = ML100K(path='{}data/ml-100k'.format(path_prefix)).ratings
elif args[1] == 'movielens_10m':
    dataset = ML10M(path='{}data/ml-10M100K'.format(path_prefix)).ratings
else:
    dataset = pd.read_csv('{}data/preprocessed_data/{}.csv'.format(path_prefix, args[1]))

dataset.name = args[1]
print(dataset.name)

item_item = ItemItem(nnbrs=10000)
user_user = UserUser(nnbrs=10000)
als_bias_mf = ALSBiasedMF(300)
funk_svd = FunkSVD(1000)
bias = Bias()
bias_svd = BiasedSVD(500)

model_list = [[item_item, 'ItemItem'],
              [user_user, 'UserUser'],
              [als_bias_mf, 'ALSBiasMf'],
              [funk_svd, 'FunkSVD'],
              [bias, 'Bias'],
              [bias_svd, 'BiasSVD']]

for default_model, model_name in model_list:
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

        fallback = Bias()
        model = Fallback(default_model, fallback)

        st = time.time()
        model.fit(train_split)
        time_taken = time.time() - st

        predictions = model.predict(X_test_split)
        root_mean_square_errors = np.append(root_mean_square_errors,
                                            rmse(predictions, y_test_split, missing='ignore'))
        np.savetxt('{}output/rmse/{}_{}_{}'.format(path_prefix, model_name, dataset.name, i), root_mean_square_errors)

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

    filer.save_predicions_to_csv_file('{}output/predictions/{}_{}_default_predictions'.format(path_prefix,
                                                                                              model_name,
                                                                                              dataset.name),
                                      dataset,
                                      prediction_dataframe)

    datapoint = pd.DataFrame([(dataset.name, model_name, root_mean_square_errors.mean(),
                              'RecSys', time_taken, time.time())],
                             columns=['dataset', 'model', 'rmse', 'category', 'time_in_sec', 'timestamp'])
    filer.append_data_to_csv(datapoint, '{}output/default_config_results.csv'.format(path_prefix))

