from lenskit.algorithms import user_knn
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter


class UserUser(user_knn.UserUser):
    def __init__(self, nnbrs, **kwargs):
        super().__init__(nnbrs=nnbrs, **kwargs)

    @staticmethod
    def get_default_configspace_hyperparameters():
        """
            return default configuration spaces for hyperparameter
        """
        user_user_min_nbrs = UniformIntegerHyperparameter('user_user_min_nbrs', lower=1, upper=50, default_value=1)
        user_user_min_sim = UniformFloatHyperparameter('user_user_min_sim', lower=0, upper=0.1, default_value=0)

        return [user_user_min_nbrs, user_user_min_sim]
