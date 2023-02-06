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
from typing import Union
from lenskit.algorithms import Recommender


def get_model_from_cs(cs: ConfigurationSpace,
                      feedback: str,
                      fallback_model=Bias(),
                      random_state: int = 42) -> Union[Recommender, Predictor]:
    """ builds a Predictor model defined in ConfigurationSpace

        Parameters
        ----------
        cs : ConfigurationSpace
            configuration space containing information to build a model
        feedback : str
            feedback type, either 'explicit' or 'implicit'
        fallback_model: Predictor
            fallback algorithm to use in case of missing values
        random_state: int
            random state to use

        Returns
        ----------
        fallback_algo : Predictor
            Predictor build with the config_space information
    """

    # check if feedback value is valid
    if (feedback != 'explicit') and (feedback != 'implicit'):
        raise ValueError("Unknown feedback type: {}".format(feedback))

    algo_name = cs.get('algo')
    config = {key.replace("{}:".format(algo_name), ""): value for key, value in cs.items()}
    del config['algo']

    # ItemItem
    if algo_name == 'ItemItem':
        model = ItemItem(feedback=feedback, **config)
    # UserUser
    elif algo_name == 'UserUser':
        model = UserUser(feedback=feedback, **config)
    # FunkSVD
    elif algo_name == 'FunkSVD':
        model = FunkSVD(random_state=random_state, **config)
    # BiasedSVD
    elif algo_name == 'BiasedSVD':
        model = BiasedSVD(**config)
    # ALSBiasedMF
    elif algo_name == 'ALSBiasedMF':
        reg_touple = (float(config['ureg']), float(config['ireg']))
        del config['ureg']
        del config['ireg']
        model = ALSBiasedMF(reg=reg_touple, rng_spec=random_state, **config)
    # Biased
    elif algo_name == 'Bias':
        damping_touple = (config['user_damping'], config['item_damping'])
        del config['user_damping']
        del config['item_damping']
        model = Bias(damping=damping_touple, **config)
    # ImplicitMF
    elif algo_name == 'ImplicitMF':
        reg_touple = (config['ureg'], config['ireg'])
        del config['ureg']
        del config['ireg']
        model = ImplicitMF(reg=reg_touple, rng_spec=random_state, **config)
    else:
        raise ValueError("Unknown algorithm: {}".format(algo_name))

    # define fallback algorithm
    fallback = fallback_model

    if feedback == 'explicit':
        final_model = Fallback(fallback, model)
    if feedback == 'implicit':
        final_model = Recommender.adapt(model)

    return final_model
