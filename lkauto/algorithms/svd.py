from lenskit.algorithms import svd
from ConfigSpace import Integer, Float, ConfigurationSpace


class BiasedSVD(svd.BiasedSVD):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace():
        """
               return default configurationspace
        """
        features = Integer('features', bounds=(5, 500), default=100)
        damping = Float('damping', bounds=(0.0, 25.5), default=5.0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, damping])

        return cs
