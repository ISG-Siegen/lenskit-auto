from lenskit.algorithms import bias
from ConfigSpace import UniformFloatHyperparameter, ConfigurationSpace


class Bias(bias.Bias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_default_configspace():
        """
            return default configurationspace
        """
        item_damping = UniformFloatHyperparameter('item_damping', lower=0, upper=1.0e+6, default_value=0, log=True)
        user_damping = UniformFloatHyperparameter('user_damping', lower=0, upper=1.0e+6, default_value=0, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([item_damping, user_damping])

        return cs
