from ConfigSpace import ConfigurationSpace
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF as ALSBiasedMF
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.basic import Fallback
from lenskit.algorithms import Predictor
from lenskit.algorithms import Recommender


def get_explicit_model_from_cs(config_space: ConfigurationSpace) -> Predictor:
    """ builds a Predictor model defined in ConfigurationSpace

        Parameters
        ----------
        config_space : ConfigurationSpace
            configuration space containing information to build a model

        Returns
        ----------
        fallback_algo : Predictor
            Predictor build with the config_space information
    """
    algo = None

    current_algo = config_space.get('regressor')

    # ItemItem
    if current_algo == 'ItemItem':
        algo = ItemItem(nnbrs=10000, **dict(config_space), feedback='explicit')

    # UserUser
    if current_algo == 'UserUser':
        algo = UserUser(nnbrs=10000, **dict(config_space), feedback='explicit')

    # ALSBiasedMF
    if current_algo == 'ALSBiasedMF':
        ureg = config_space['biased_mf_ureg']
        ireg = config_space['biased_mf_ireg']
        reg_touple = (ureg, ireg)
        algo = ALSBiasedMF(features=config_space['biased_mf_features'],
                           reg=reg_touple,
                           damping=config_space['biaseed_mf_damping'],
                           rng_spec=42)

    # Biased
    if current_algo == 'Bias':
        items_damping = config_space['bias_item_damping']
        users_damping = config_space['bias_user_damping']
        damping_touple = (users_damping, items_damping)
        algo = Bias(damping=damping_touple)

    # FunkSVD
    if current_algo == 'FunkSVD':
        algo = FunkSVD(features=config_space['funk_svd_features'],
                       lrate=config_space['funk_svd_lrate'],
                       reg=config_space['funk_svd_reg'],
                       damping=config_space['funk_svd_damping'],
                       random_state=42)

    # BiasedSVD
    if current_algo == 'BiasedSVD':
        algo = BiasedSVD(features=1000,
                         damping=config_space['bias_svd_damping'])

    # define fallback algorithm
    fallback = Bias()
    fallback_algo = Fallback(algo, fallback)

    return fallback_algo


def get_implicit_recommender_from_cs(config_space: ConfigurationSpace, random_state=None) -> Recommender:
    """ builds a Recommender model defined in ConfigurationSpace

           Parameters
           ----------
           config_space : ConfigurationSpace
               configuration space containing information to build a model

            Returns
            ----------
            fallback_algo : Recommender
                Recommender build with the config_space information
       """
    algo = None

    current_algo = config_space.get('regressor')

    # ItemItem
    if current_algo == 'ItemItem':
        algo = ItemItem(nnbrs=10000, **dict(config_space))

    # UserUser
    if current_algo == 'UserUser':
        algo = UserUser(nnbrs=10000,
                        min_nbrs=config_space['user_user_min_nbrs'],
                        min_sim=config_space['user_user_min_sim'],
                        feedback='implicit')

    # FunkSVD
    if current_algo == 'FunkSVD':
        algo = FunkSVD(features=config_space['funk_svd_features'],
                       lrate=config_space['funk_svd_lrate'],
                       reg=config_space['funk_svd_reg'],
                       damping=config_space['funk_svd_damping'],
                       random_state=random_state)

    # ImplicitMF
    if current_algo == 'ImplicitMF':
        ureg = config_space['implicit_mf_ureg']
        ireg = config_space['implicit_mf_ireg']
        reg_touple = (ureg, ireg)
        algo = ImplicitMF(features=config_space['implicit_mf_features'],
                          reg=reg_touple,
                          weight=config_space['implicit_mf_weight'])

    # BiasedSVD
    if current_algo == 'BiasedSVD':
        algo = BiasedSVD(features=config_space[' bias_svd_features'],
                         damping=config_space['bias_svd_damping'])

    # define fallback algorithm
    fallback = Bias()
    fallback_algo = Fallback(algo, fallback)

    # transorm Predictor to Recommender
    recommender_algo = Recommender.adapt(fallback_algo)


    return recommender_algo


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
                           weight=config_space['implicit_mf_weight'])

    if current_model == 'BiasedSVD':
        model = BiasedSVD(features=config_space['bias_svd_features'],
                          damping=config_space['bias_svd_damping'])

    recommender = Recommender.adapt(model)

    return recommender

