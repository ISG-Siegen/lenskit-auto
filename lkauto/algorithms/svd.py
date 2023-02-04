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

        """
        The algorithm is based on the scikit TruncatedSVD algorithm and the hyperparameters are
        choosen based around the default values of LensKit. No further explaination could be distracted
        from scikit or the original paper.
        """
        features = Integer('features', bounds=(2, 10000), default=1000, log=True)  # No default values given
        damping = Float('damping', bounds=(0.0, 1000), default=5.0, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, damping])

        return cs
