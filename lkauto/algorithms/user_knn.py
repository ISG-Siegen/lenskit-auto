from lenskit.algorithms import user_knn
from ConfigSpace import Integer, Float
from ConfigSpace import ConfigurationSpace


class UserUser(user_knn.UserUser):
    def __init__(self, nnbrs, **kwargs):
        super().__init__(nnbrs=nnbrs, **kwargs)

    @staticmethod
    def get_default_configspace():
        """
            return default configuration spaces for hyperparameter
        """
        min_nbrs = Integer('min_nbrs', bounds=(1, 50), default=1)
        min_sim = Float('min_sim', bounds=(0, 0.1), default=0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([min_nbrs, min_sim])

        return cs
