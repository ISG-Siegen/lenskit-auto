from lenskit.algorithms import bias
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter


class Bias(bias.Bias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_default_configspace_hyperparameters():
        """
            return default configuration spaces for hyperparameter
        """
        bias_item_damping = UniformFloatHyperparameter('bias_item_damping', lower=1.0, upper=25, default_value=5)
        bias_user_damping = UniformFloatHyperparameter('bias_user_damping', lower=1.0, upper=25, default_value=5)

        return [bias_item_damping, bias_user_damping]
