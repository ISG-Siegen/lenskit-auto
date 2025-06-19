from typing import Union

from ConfigSpace import ConfigurationSpace

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
    if algo_name == 'ItemKNNScorer':
        # model = ItemKNNScorer(feedback=feedback, **config)
        model = ItemKNNScorer()
    # UserUser
    elif algo_name == 'UserKNNScorer':
        # model = UserKNNScorer(feedback=feedback, **config)
        model = UserKNNScorer()
    # FunkSVD
    elif algo_name == 'FunkSVDScorer':
        # model = FunkSVDScorer(random_state=random_state, **config)
        model = FunkSVDScorer()
    # BiasedSVD
    elif algo_name == 'BiasedSVDScorer':
        # model = BiasedSVDScorer(**config)
        model = BiasedSVDScorer()
    # ALSBiasedMF
    elif algo_name == 'BiasedMFScorer':
        reg_touple = (float(config.pop('ureg')), float(config.pop('ireg')))
        # del config['ureg']
        # del config['ireg']
        # model = BiasedMFScorer(reg=reg_touple, rng_spec=random_state, **config)
        model = BiasedMFScorer()
    # Biased
    elif algo_name == 'BiasScorer':
        damping_touple = (config.pop('user_damping'), config.pop('item_damping'))
        # del config['user_damping']
        # del config['item_damping']
        # model = BiasScorer(damping=damping_touple, **config)
        model = BiasScorer()
    # ImplicitMF
    elif algo_name == 'ImplicitMFScorer':
        reg_touple = (float(config.pop('ureg')), float(config.pop('ireg')))
        # del config['ureg']
        # del config['ireg']
        # model = ImplicitMFScorer(reg=reg_touple, rng_spec=random_state, **config)
        model = ImplicitMFScorer()
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
