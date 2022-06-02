from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF as ALSBiasedMF
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.basic import Fallback
from lenskit.metrics.predict import rmse
import lenskit.crossfold as xf
import numpy as np


class ExplicitEvaler:

    def __init__(self, train, random_seed=None, folds=None):
        self.train = train
        self.random_seed = random_seed
        self.folds = folds

    @staticmethod
    def get_explicit_model_from_cs(config_space):
        model = None

        current_model = config_space.get('regressor')

        # ItemItem
        if current_model == 'ItemItem':
            model = ItemItem(nnbrs=10000,
                             min_nbrs=config_space['item_item_min_nbrs'],
                             min_sim=config_space['item_item_min_sim'],
                             feedback='explicit')
        # UserUser
        if current_model == 'UserUser':
            model = UserUser(nnbrs=10000,
                             min_nbrs=config_space['user_user_min_nbrs'],
                             min_sim=config_space['user_user_min_sim'],
                             feedback='explicit')
        # ALSBiasedMF
        if current_model == 'ALSBiasedMF':
            ureg = config_space['biased_mf_ureg']
            ireg = config_space['biased_mf_ireg']
            reg_touple = (ureg, ireg)
            model = ALSBiasedMF(features=config_space['biased_mf_features'],
                                reg=reg_touple,
                                damping=config_space['biaseed_mf_damping'])

        # Biased
        if current_model == 'Bias':
            items_damping = config_space['bias_item_damping']
            users_damping = config_space['bias_user_damping']
            damping_touple = (users_damping, items_damping)
            model = Bias(damping=damping_touple)

        # FunkSVD
        if current_model == 'FunkSVD':
            model = FunkSVD(features=config_space['funk_svd_features'],
                            lrate=config_space['funk_svd_lrate'],
                            reg=config_space['funk_svd_reg'],
                            damping=config_space['funk_svd_damping'])

        # BiasedSVD
        if current_model == 'BiasedSVD':
            model = BiasedSVD(features=1000,
                              damping=config_space['bias_svd_damping'])

        return model

    # TODO: make metric depending on a metiric parameter
    def evaluate_explicit(self, configuration_space):
        root_mean_square_errors = np.array([])
        config_space_model = self.get_explicit_model_from_cs(configuration_space)
        fallback = Bias()
        model = Fallback(config_space_model, fallback)

        for i, tp in enumerate(xf.partition_rows(self.train, self.folds, rng_spec=self.random_seed)):
            train_vaidation_split = tp.train.copy()
            X_test_vaidation_split = tp.test.copy()
            X_test_vaidation_split.drop('rating', inplace=True, axis=1)
            y_test_validation_split = tp.test.copy()
            y_test_validation_split = y_test_validation_split[['rating']].iloc[:, 0]

            model.fit(train_vaidation_split)
            predictions = model.predict(X_test_vaidation_split)

            root_mean_square_errors = np.append(root_mean_square_errors,
                                                rmse(predictions, y_test_validation_split, missing='ignore'))

        return root_mean_square_errors.mean()
