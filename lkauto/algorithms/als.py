from lenskit.algorithms import als
from ConfigSpace import Integer, Float, ConfigurationSpace, Categorical, Uniform


class ImplicitMF(als.ImplicitMF):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        """
              return default configurationspace
        """

        """
        The Agorithm is based on the https://ieeexplore.ieee.org/document/4781121 papaer.
        The paper does not give clear hints on how to set up the hyperparameters best.
        Therefore we stick to the ranges from LensKits other Matrix Factorization algorithm: BiasedMF
        """
        features = Integer('features', bounds=(5, 10000), default=1000)  # No default values given
        ureg = Float('ureg', bounds=(0.01, 0.1), default=0.1)
        ireg = Float('ireg', bounds=(0.01, 0.1), default=0.1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, ureg, ireg])

        return cs


class BiasedMF(als.BiasedMF):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

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
        features = Integer('features', bounds=(2, 10000), default=1000, log=True)  # No default value given

        """
        The authors of the original ALS paper set the range of the regularization hyperparameter to from 0.03 - 0.065.
        https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32
        Therefore we set the lower bound of the two regularization parameters (ureg and ireg) to 0.065.
        LensKit sets the default regularization hyperparameter to 0.1 Therefore we set the upper bound of the two
        regularization parameters (ureg and ireg) to 0.1.
        """
        ureg = Float('ureg', bounds=(0.01, 0.1), default=0.1)
        ireg = Float('ireg', bounds=(0.01, 0.1), default=0.1)

        """
        The damping hyperparameter en- or disables a damping factor.
        In the future we may want to tune the damping values as well.
        """
        bias = Categorical('bias', items=[True, False], default=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, ureg, ireg, bias])

        return cs
