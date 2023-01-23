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
        item_damping = UniformFloatHyperparameter('item_damping', lower=1.0, upper=25, default_value=5)
        user_damping = UniformFloatHyperparameter('user_damping', lower=1.0, upper=25, default_value=5)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([item_damping, user_damping])

        return cs
