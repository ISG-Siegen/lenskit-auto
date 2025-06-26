from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter


class ImplicitMF(ImplicitMFScorer):
    def __init__(self, features, feedback="implicit", **kwargs):
        super().__init__(feedback=feedback, features=features, **kwargs)
        self.features = features  # store the features as an instance variable for testing

    @staticmethod
    def get_default_configspace(**kwargs):
        """
              return default configurationspace
        """
        features = UniformIntegerHyperparameter('features', lower=5, upper=10000, default_value= 1000, log=True)
        ureg = UniformFloatHyperparameter('ureg', lower=0.01, upper=0.1, default_value=0.1, log=True)
        ireg = UniformFloatHyperparameter('ireg', lower=0.01, upper=0.1, default_value=0.1, log=True)

        cs = ConfigurationSpace()
        cs.add([features, ureg, ireg])
        return cs


class BiasedMF(BiasedMFScorer):
    def __init__(self, features, feedback="explicit", **kwargs):
        super().__init__(features=features, **kwargs)
        self.feedback = feedback
        self.features = features # store the features as an instance variable for testing

    @staticmethod
    def get_default_configspace(**kwargs):
        """
               return default configuration spaces for hyperparameter
        """

        """
        The authors of the original ALS paper (https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32) stated:
        The most important discovery we made is that ALS-WR never overfits the data if we either increase
        the number of iterations or the number of hidden features.
        The paper stated that the improvement of the performance maximized around 1000 features.
        Therefore, we will set the upper bound and the default value of features to 10000.
        Since the authors just evaluated on one larger dataset, we still allow smaller and larger feature numbers
        but set the default value to 1000.
        """
        #features = Integer('features', bounds=(2, 10000), default=1000, log=True)  # No default value given
        # no default value given but we set the default value to 1000???
        features = UniformIntegerHyperparameter('features', lower=2, upper=10000, default_value=1000, log=True)
        """
        The authors of the original ALS paper set the range of the regularization hyperparameter to from 0.03 - 0.065.
        https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32
        Therefore we set the lower bound of the two regularization parameters (ureg and ireg) to 0.065.
        LensKit sets the default regularization hyperparameter to 0.1 Therefore we set the upper bound of the two
        regularization parameters (ureg and ireg) to 0.1.
        """
        ureg = UniformFloatHyperparameter('ureg', lower=0.01, upper=0.1, default_value=0.1, log=True)
        ireg = UniformFloatHyperparameter('ireg', lower=0.01, upper=0.1, default_value=0.1, log=True)


        """
        The damping hyperparameter en- or disables a damping factor.
        In the future we may want to tune the damping values as well.
        """
        bias = CategoricalHyperparameter('bias', choices=[True, False], default_value=True)

        cs = ConfigurationSpace()
        cs.add([features, ureg, ireg, bias])
        return cs
