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
        # BiasedSVD configspace provides 'damping' (single value), default is 5
        damping = float(config.get('damping', 5.0))  # LensKit default is 5
        cfg = BiasedSVDConfig(
            damping=damping
        )
        model = BiasedSVDScorer(config=cfg)
    # ALSBiasedMF
    elif algo_name == 'ALSBiasedMF':
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
        # build damping dict only if parameters are provided in config
        # if not provided, BiasScorer will use LensKit's default (0.0)
        damping_dict = {}
        if 'user_damping' in config:
            damping_dict['user'] = float(config['user_damping'])
        if 'item_damping' in config:
            damping_dict['item'] = float(config['item_damping'])

        # pass damping only if we have at least one value
        if damping_dict:
            model = BiasScorer(damping=damping_dict)
        else:
            # use LensKit's default damping (0.0)
            model = BiasScorer()
    # ImplicitMF
    elif algo_name == 'ImplicitMF':
        cfg = ImplicitMFConfig(
            embedding_size=int(config.get('features', 50)),
            user_embeddings=True if config.get('bias', True) else 'prefer',
        )
        model = ImplicitMFScorer(config=cfg)
    else:
        raise ValueError("Unknown algorithm: {}".format(algo_name))

    final_model = model
    return final_model
