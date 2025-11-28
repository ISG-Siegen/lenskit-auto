from typing import Union

from ConfigSpace import ConfigurationSpace

from lenskit.als import BiasedMFScorer, BiasedMFConfig
from lenskit.data.types import UIPair
from lenskit.als import ImplicitMFScorer, ImplicitMFConfig
from lenskit.basic import BiasScorer
from lenskit.basic.bias import BiasConfig #changed to basic.BiasConfig in newer versions
from lenskit.funksvd import FunkSVDScorer, FunkSVDConfig
from lenskit.knn import ItemKNNScorer, ItemKNNConfig
from lenskit.sklearn.svd import BiasedSVDScorer, BiasedSVDConfig
from lenskit.knn import UserKNNScorer, UserKNNConfig
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
    print("Algo:", algo_name)
    print("Incoming config:", config)
    del config['algo']

    # ItemItem
    if algo_name == 'ItemItem':
        cfg = ItemKNNConfig(feedback=feedback, **config)
        model = ItemKNNScorer(cfg)
    # UserUser
    elif algo_name == 'UserUser':
        cfg = UserKNNConfig(feedback=feedback, **config)
        model = UserKNNScorer(cfg)
    # FunkSVD
    elif algo_name == 'FunkSVD':
        cfg = FunkSVDConfig(feedback=feedback, **config)
        model = FunkSVDScorer(cfg)
    # BiasedSVD
    elif algo_name == 'BiasedSVD':
        damping_dict = {
            'user': float(config.get('user_damping', 5.0)),  # default 5
            'item': float(config.get('item_damping', 5.0))  # default 5
        }
        model = BiasedSVDScorer(**damping_dict)
    # ALSBiasedMF
    elif algo_name == 'ALSBiasedMF':
        reg = UIPair(
            user=config.get('ureg', 0.1),
            item=config.get('ireg', 0.1)
        )
        cfg = BiasedMFConfig(feedback=feedback, **config)
        model = BiasedMFScorer(cfg)
    # Biased
    elif algo_name == 'Bias':
        user_damping = float(config.get('user_damping'))
        item_damping = float(config.get('item_damping'))

        # New API: pass as dict instead of tuple
        damping_dict = {'user': user_damping, 'item': item_damping}
        cfg = BiasConfig(damping=damping_dict)
        model = BiasScorer(cfg)
    # ImplicitMF
    elif algo_name == 'ImplicitMF':
        cfg = ImplicitMFConfig(feedback=feedback, **config)
        model = ImplicitMFScorer(cfg)
    else:
        raise ValueError("Unknown algorithm: {}".format(algo_name))

    final_model = model
    # final_model.config = type(final_model.config) #added this to test something.
    return final_model
