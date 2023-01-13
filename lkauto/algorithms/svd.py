from lenskit.algorithms import svd
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter


class BiasedSVD(svd.BiasedSVD):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace_hyperparameters():
        """
               return default configuration spaces for hyperparameter
        """
        bias_svd_features = UniformIntegerHyperparameter('bias_svd_features', lower=5, upper=500)
        bias_svd_damping = UniformFloatHyperparameter('bias_svd_damping', lower=0.0, upper=25.5, default_value=5)

        return [bias_svd_features, bias_svd_damping]
