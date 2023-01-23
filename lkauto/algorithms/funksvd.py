from lenskit.algorithms import funksvd
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter, ConfigurationSpace


class FunkSVD(funksvd.FunkSVD):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace():
        """
               return default configurationspace
        """
        features = UniformIntegerHyperparameter('features', lower=100, upper=500)
        lrate = UniformFloatHyperparameter('lrate', lower=0.0, upper=0.1, default_value=0.001)
        reg = UniformFloatHyperparameter('reg', lower=0.0, upper=0.1, default_value=0.0015)
        damping = UniformFloatHyperparameter('damping', lower=0.0, upper=25.0, default_value=5)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, lrate, reg, damping])

        return cs
