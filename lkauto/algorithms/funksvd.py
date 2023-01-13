from lenskit.algorithms import funksvd
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter


class FunkSVD(funksvd.FunkSVD):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace_hyperparameters():
        """
               return default configuration spaces for hyperparameter
        """
        funk_svd_features = UniformIntegerHyperparameter('funk_svd_features', lower=100, upper=500)
        funk_svd_lrate = UniformFloatHyperparameter('funk_svd_lrate', lower=0.0, upper=0.1, default_value=0.001)
        funk_svd_reg = UniformFloatHyperparameter('funk_svd_reg', lower=0.0, upper=0.1, default_value=0.0015)
        funk_svd_damping = UniformFloatHyperparameter('funk_svd_damping', lower=0.0, upper=25.0, default_value=5)

        return [funk_svd_features, funk_svd_lrate, funk_svd_reg, funk_svd_damping]
