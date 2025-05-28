from typing import Union

from ConfigSpace import ConfigurationSpace
# from lenskit.algorithms import Predictor
# from lenskit.algorithms import Recommender
# from lenskit.algorithms.als import BiasedMF
# from lenskit.algorithms.als import ImplicitMF
# from lenskit.algorithms.basic import Fallback
# from lenskit.algorithms.bias import Bias
# from lenskit.algorithms.funksvd import FunkSVD
# from lenskit.algorithms.item_knn import ItemItem
# from lenskit.algorithms.svd import BiasedSVD
# from lenskit.algorithms.user_knn import UserUser

from lenskit.als import BiasedMFScorer
from lenskit.als import ImplicitMFScorer
#fallback missing
from lenskit.basic import BiasScorer
from lenskit.funksvd import FunkSVDScorer
from lenskit.knn import ItemKNNScorer
from lenskit.sklearn.svd import BiasedSVDScorer
from lenskit.knn import UserKNNScorer
# from lenskit.scored import Scorer
from lenskit.pipeline import Component


def get_model_from_cs(cs: ConfigurationSpace,
                      feedback: str,
                      # fallback_model=Bias(),
                      random_state: int = 42) ->Union[
                          ItemKNNScorer,
                          UserKNNScorer,
                          FunkSVDScorer,
                          BiasedSVDScorer,
                          BiasedMFScorer,
                          BiasScorer,
                          ImplicitMFScorer]: #Union[Recommender, Predictor]:
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

    # get algorithm name
    algo_name = cs.get('algo')
    config = {key.replace("{}:".format(algo_name), ""): value for key, value in cs.items()}
    del config['algo']

    # ItemItem
    if algo_name == 'ItemItem':
        model = ItemKNNScorer(feedback=feedback, **config)
    # UserUser
    elif algo_name == 'UserUser':
        model = UserKNNScorer(feedback=feedback, **config)
    # FunkSVD
    elif algo_name == 'FunkSVD':
        model = FunkSVDScorer(random_state=random_state, **config)
    # BiasedSVD
    elif algo_name == 'BiasedSVD':
        model = BiasedSVDScorer(**config)
    # ALSBiasedMF
    elif algo_name == 'ALSBiasedMF':
        reg_touple = (float(config.pop('ureg')), float(config.pop('ireg')))
        # del config['ureg']
        # del config['ireg']
        model = BiasedMFScorer(reg=reg_touple, rng_spec=random_state, **config)
    # Biased
    elif algo_name == 'Bias':
        damping_touple = (config.pop('user_damping'), config.pop('item_damping'))
        # del config['user_damping']
        # del config['item_damping']
        model = BiasScorer(damping=damping_touple, **config)
    # ImplicitMF
    elif algo_name == 'ImplicitMF':
        reg_touple = (float(config.pop('ureg')), float(config.pop('ireg')))
        # del config['ureg']
        # del config['ireg']
        model = ImplicitMFScorer(reg=reg_touple, rng_spec=random_state, **config)
    else:
        raise ValueError("Unknown algorithm: {}".format(algo_name))

    # define fallback algorithm
    # fallback = fallback_model

    # define final model
    # if feedback == 'explicit':
    #     final_model = Fallback(model, fallback)
    # if feedback == 'implicit':
    #     final_model = Recommender.adapt(model)

    final_model = model
    return final_model
