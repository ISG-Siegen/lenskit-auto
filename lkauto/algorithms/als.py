from lenskit.algorithms import als
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter


class ImplicitMF(als.ImplicitMF):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace_hyperparameters():
        """
              return default configuration spaces for hyperparameter
        """
        implicit_mf_features = UniformIntegerHyperparameter('implicit_mf_features', lower=5, upper=300)
        implicit_mf_ureg = UniformFloatHyperparameter('implicit_mf_ureg', lower=1.0e-6, upper=5,
                                                      default_value=0.1)
        implicit_mf_ireg = UniformFloatHyperparameter('implicit_mf_ireg', lower=1.0e-6, upper=5,
                                                      default_value=0.1)
        implicit_mf_weight = UniformFloatHyperparameter('implicit_mf_weight', lower=10.0, upper=200.0,
                                                        default_value=40.0)

        return [implicit_mf_features, implicit_mf_ureg, implicit_mf_ireg, implicit_mf_weight]


class BiasedMF(als.BiasedMF):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace_hyperparameters():
        """
               return default configuration spaces for hyperparameter
        """

        biased_mf_features = UniformIntegerHyperparameter('biased_mf_features', lower=5, upper=300)
        biased_mf_ureg = UniformFloatHyperparameter('biased_mf_ureg', lower=1.0e-6, upper=5, default_value=0.1)
        biased_mf_ireg = UniformFloatHyperparameter('biased_mf_ireg', lower=1.0e-6, upper=5, default_value=0.1)
        biaseed_mf_damping = UniformFloatHyperparameter('biaseed_mf_damping', lower=0.0, upper=25.0, default_value=5.0)

        return [biased_mf_features, biased_mf_ureg, biased_mf_ireg, biaseed_mf_damping]
