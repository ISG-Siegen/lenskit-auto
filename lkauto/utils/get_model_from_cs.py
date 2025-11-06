from typing import Union

from ConfigSpace import ConfigurationSpace

from lenskit.als import BiasedMFScorer, BiasedMFConfig
from lenskit.data.types import UIPair
from lenskit.als import ImplicitMFScorer, ImplicitMFConfig
from lenskit.basic import BiasScorer
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
    del config['algo']

    # ItemItem
    if algo_name == 'ItemItem':
        # model = ItemKNNScorer(feedback=feedback, **config)
        cfg = ItemKNNConfig(feedback=feedback,
                            max_nbrs=config.get('max_nbrs', 20),
                            min_nbrs=config.get('min_nbrs', 1),
                            min_sim=config.get('min_sim', 1e-6))
        model = ItemKNNScorer(config=cfg)
    # UserUser
    elif algo_name == 'UserUser':
        # model = UserKNNScorer(feedback=feedback, **config)
        cfg = UserKNNConfig(feedback=feedback,
                            max_nbrs=config.get('max_nbrs', 20),
                            min_nbrs=config.get('min_nbrs', 1),
                            min_sim=config.get('min_sim', 1e-6))
        model = UserKNNScorer(config=cfg)
    # FunkSVD
    elif algo_name == 'FunkSVD':
        # model = FunkSVDScorer(random_state=random_state, **config)
        cfg = FunkSVDConfig(
            embedding_size=int(config.get('features', 50)),  # default 50
            learning_rate=float(config.get('lrate', 0.001)),  # default 0.001
            regularization=float(config.get('reg', 0.015)),  # default 0.015
            damping=float(config.get('damping', 5.0)),  # default 5.0
        )
        model = FunkSVDScorer(config=cfg)
    # BiasedSVD
    elif algo_name == 'BiasedSVD':
        # model = BiasedSVDScorer(**config)
        damping_dict = {
            'user': float(config.get('user_damping', 5.0)),  # default 5
            'item': float(config.get('item_damping', 5.0))  # default 5
        }
        cfg = BiasedSVDConfig(
            damping=damping_dict
        )
        model = BiasedSVDScorer(config=cfg)
    # ALSBiasedMF
    elif algo_name == 'ALSBiasedMF':
        # reg_touple = (float(config.pop('ureg')), float(config.pop('ireg')))
        print(config.keys())  # delete later for debugging
        # del config['ureg']
        # del config['ireg']
        # model = BiasedMFScorer(reg=reg_touple, rng_spec=random_state, **config)
        reg = UIPair(
            user=config.get('ureg', 0.1),
            item=config.get('ireg', 0.1)
        )
        cfg = BiasedMFConfig(
            embedding_size=int(config.get('features', 50)),
            regularization=reg,
            user_embeddings=True if config.get('bias', True) else 'prefer',
        )
        model = BiasedMFScorer(config=cfg)
    # Biased
    elif algo_name == 'Bias':
        user_damping = float(config.get('user_damping'))
        item_damping = float(config.get('item_damping'))

        # New API: pass as dict instead of tuple
        damping_dict = {'user': user_damping, 'item': item_damping}
        model = BiasScorer(damping=damping_dict)
        # model = BiasScorer(damping=damping_dict, **config)
    # ImplicitMF
    elif algo_name == 'ImplicitMF':
        # reg_touple = (config.pop('ureg'), config.pop('ireg'))
        # model = ImplicitMFScorer(reg=reg_touple, rng_spec=random_state, **config)
        cfg = ImplicitMFConfig(
            embedding_size=int(config.get('features', 50)),
            # regularization=reg_touple,
            user_embeddings=True if config.get('bias', True) else 'prefer',
        )
        model = ImplicitMFScorer(config=cfg)
    else:
        raise ValueError("Unknown algorithm: {}".format(algo_name))

    final_model = model
    final_model.config = type(final_model.config) #added this to test something.
    return final_model
