from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF as ALSBiasedMF
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.basic import Fallback
from lenskit.algorithms.als import ImplicitMF
from lenskit import Recommender


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

    fallback = Bias()
    fallback_model = Fallback(model, fallback)

    return fallback_model


def get_implicit_recommender_from_cs(config_space, random_state=None):
    model = None

    current_model = config_space.get('regressor')

    if current_model == 'ItemItem':
        model = ItemItem(nnbrs=10000,
                         min_nbrs=config_space['item_item_min_nbrs'],
                         min_sim=config_space['item_item_min_sim'],
                         feedback='implicit')

    if current_model == 'UserUser':
        model = UserUser(nnbrs=10000,
                         min_nbrs=config_space['user_user_min_nbrs'],
                         min_sim=config_space['user_user_min_sim'],
                         feedback='implicit')

    if current_model == 'FunkSVD':
        model = FunkSVD(features=config_space['funk_svd_features'],
                        lrate=config_space['funk_svd_lrate'],
                        reg=config_space['funk_svd_reg'],
                        damping=config_space['funk_svd_damping'],
                        random_state=random_state)

    if current_model == 'ImplicitMF':
        ureg = config_space['implicit_mf_ureg']
        ireg = config_space['implicit_mf_ireg']
        reg_touple = (ureg, ireg)
        model = ImplicitMF(features=config_space['implicit_mf_features'],
                           reg=reg_touple,
                           weight=config_space['implicit_mf_damping'])

    if current_model == 'BiasedSVD':
        model = BiasedSVD(features=config_space['bias_svd_features'],
                          damping=config_space['bias_svd_damping'])

    recommender = Recommender.adapt(model)

    return recommender
