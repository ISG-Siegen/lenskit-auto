from lenskit.algorithms import als
from ConfigSpace import Integer, Float, ConfigurationSpace


class ImplicitMF(als.ImplicitMF):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace():
        """
              return default configurationspace
        """
        features = Integer('features', bounds=(5, 300), default=100)
        ureg = Float('ureg', bounds=(1.0e-6, 5), default=0.1)
        ireg = Float('ireg', bounds=(1.0e-6, 5), default=0.1)
        weight = Float('weight', bounds=(10.0, 200.0), default=40.0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, ureg, ireg, weight])

        return cs


class BiasedMF(als.BiasedMF):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace():
        """
               return default configuration spaces for hyperparameter
        """

        features = Integer('features', bounds=(5, 300), default=100)
        ureg = Float('ureg', bounds=(1.0e-6, 5), default=0.1)
        ireg = Float('ireg', bounds=(1.0e-6, 5), default=0.1)
        damping = Float('damping', bounds=(0.0, 25.0), default=5.0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, ureg, ireg, damping])

        return cs
