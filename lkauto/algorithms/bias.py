from lenskit.basic.bias import BiasScorer
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter


class Bias(BiasScorer):
    def __init__(self, feedback="explicit", **kwargs):
        super().__init__(**kwargs)

        self.feedback = feedback

    @staticmethod
    def get_default_configspace(number_item: int, number_user: int):
        """
            return default configurationspace
        """

        """
        LensKit does not give any hint on how to setup the damping values for the Bias algorithm.
        Therefore we evaluated the algorithms performance on 70 datasets to come up with the following ranges
        """
        # lower bound 0 does not work because of log=True
        item_damping = UniformFloatHyperparameter('item_damping', lower=1e-5*number_item, upper=85*number_item,
                                                  default_value=0.0025*number_item, log=True)
        user_damping = UniformFloatHyperparameter('user_damping', lower=1e-5*number_user, upper=85*number_user,
                                                  default_value=0.0025*number_user, log=True)

        cs = ConfigurationSpace()
        cs.add([item_damping, user_damping])

        return cs
