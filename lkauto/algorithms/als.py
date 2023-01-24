from lenskit.algorithms import als
from ConfigSpace import Integer, Float, ConfigurationSpace, Categorical, Constant


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

        """
        The authors of the original ALS paper (https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32) stated:

        The most important discovery we made is that ALS-WR never overfits the data if we either increase
        the number of iterations or the number of hidden features.

        The paper stated that the improvement of the performance maximized around 1000 features.

        Therefore we set the upper bound of the features hyperparameter to 1000.
        """
        features = Integer('features', bounds=(2, 1000))  # No default value given

        """
        The authors of the original ALS paper set the range of the regularization hyperparameter to from 0.03 - 0.065.
        https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32
        Therefore we set the lower bound of the two regularization parameters (ureg and ireg) to 0.065.
        LensKit sets the default regularization hyperparameter to 0.1 Therefore we set the upper bound of the two
        regularization parameters (ureg and ireg) to 0.1.
        """
        ureg = Float('ureg', bounds=(0.065, 0.1), default=0.1)
        ireg = Float('ireg', bounds=(0.065, 0.1), default=0.1)

        # FIXME: Add explanation for damping hyperparameter
        """
        The damping hyperparameter describes the damping factor for the underlying bias.
        """
        user_damping = Float('user_damping', bounds=(0.0, 25.0), default=5.0)
        item_damping = Float('item_damping', bounds=(0.0, 25.0), default=5.0)

        """
        The method hyperparameter decides between two solvers for the optimization step.
        The "lu" (LU-decomposition) solver is taken from the original ALS paper
        (https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32).
        The "cd" (coordinate descent) solver is described in the https://dl.acm.org/doi/10.1145/2043932.2043987 paper.
        LensKit uses the "cd" solver by default. Therefore LensKit-Auto sticks to this decision.
        """
        method = Categorical('method', items=['cd', 'lu'], default='cd')

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, ureg, ireg, user_damping, item_damping, method])

        return cs
