from typing import Union

from ConfigSpace import ConfigurationSpace

from lenskit.als import BiasedMFScorer
from lenskit.als import ImplicitMFScorer
from lenskit.basic import BiasScorer
from lenskit.funksvd import FunkSVDScorer
from lenskit.knn import ItemKNNScorer
from lenskit.sklearn.svd import BiasedSVDScorer
from lenskit.knn import UserKNNScorer
# from lenskit.scored import Scorer


def get_model_from_cs(cs: ConfigurationSpace,
                      feedback: str,
                      # fallback_model=Bias(),
                      random_state: int = 42) -> Union[
                          ItemKNNScorer,
                          UserKNNScorer,
                          FunkSVDScorer,
                          BiasedSVDScorer,
                          BiasedMFScorer,
                          BiasScorer,
                          ImplicitMFScorer]:
    """ builds a Predictor model defined in ConfigurationSpace

        Parameters
        ----------
        cs : ConfigurationSpace
            configuration space containing information to build a model
        feedback : str
            feedback type, either 'explicit' or 'implicit'
        random_state: int
            random state to use

        Returns
        ----------
        model: Union[ItemKNNScorer, UserKNNScorer, FunkSVDScorder, BiasedSVDScorer, BiasedMFScorer, BiasScorer, ImplicitMFScorer]
            Model build with the config_space information
    """

    # check if feedback value is valid
    if (feedback != 'explicit') and (feedback != 'implicit'):
        raise ValueError("Unknown feedback type: {}".format(feedback))

    # get algorithm name
    algo_name = cs.get('algo')
    # config = {key.replace("{}:".format(algo_name), ""): value for key, value in cs.items()} #changed : to ; in .replace
    config = {
        key.replace(f"{algo_name}:", "").replace(f"{algo_name};", ""): value
        for key, value in cs.items()
    }
    del config['algo']

    # ItemItem
    if algo_name == 'ItemItem':
        model = ItemKNNScorer(feedback=feedback, **config)
        # model = ItemKNNScorer(feedback=feedback)
    # UserUser
    elif algo_name == 'UserUser':
        model = UserKNNScorer(feedback=feedback, **config)
        # model = UserKNNScorer(feedback=feedback)
    # FunkSVD
    elif algo_name == 'FunkSVD':
        model = FunkSVDScorer(random_state=random_state, **config)
        # model = FunkSVDScorer(feedback=feedback)
    # BiasedSVD
    elif algo_name == 'BiasedSVD':
        model = BiasedSVDScorer(**config)
        # model = BiasedSVDScorer(feedback=feedback)
    # ALSBiasedMF
    elif algo_name == 'ALSBiasedMF':
        reg_touple = (float(config.pop('ureg')), float(config.pop('ireg')))
        print(config.keys()) #delete later for debugging
        # del config['ureg']
        # del config['ireg']
        model = BiasedMFScorer(reg=reg_touple, rng_spec=random_state, **config)
        # model = BiasedMFScorer(feedback=feedback)
    # Biased
    elif algo_name == 'Bias':
        user_damping = float(config.pop('user_damping'))
        item_damping = float(config.pop('item_damping'))

        # New API: pass as dict instead of tuple
        damping_dict = {'user': user_damping, 'item': item_damping}
        model = BiasScorer(damping=damping_dict, **config)
    # ImplicitMF
    elif algo_name == 'ImplicitMF':
        reg_touple = (float(config.pop('ureg')), float(config.pop('ireg')))
        # del config['ureg']
        # del config['ireg']
        model = ImplicitMFScorer(reg=reg_touple, rng_spec=random_state, **config)
        # model = ImplicitMFScorer(feedback=feedback)
    else:
        raise ValueError("Unknown algorithm: {}".format(algo_name))

    final_model = model
    return final_model
