from lenskit.algorithms import user_knn
from ConfigSpace import Integer, Float, Constant
from ConfigSpace import ConfigurationSpace


class UserUser(user_knn.UserUser):
    def __init__(self, nnbrs, **kwargs):
        super().__init__(nnbrs=nnbrs, **kwargs)

    @staticmethod
    def get_default_configspace():
        """
            return default configuration spaces for hyperparameter
        """
        """
        The nnbrs hyperparameter is set to 10000 in LensKit-Auto. Generally speaking, the higher the nnbrs
        hyperparameter value, the better the performance. But the computational cost will increase
        exponentially by increasing the nnbrs hyperparameter value. 10000 is a reasonable value for nnbrs
        hyperparameter since it has relatively good performance and is still able
        to run in a reasonable amount of time.
        """
        nnbrs = Integer('nnbrs', bounds=(1, 10000), default=10000, log=True)  # No default value given by LensKit

        """
        The min_sim hyperparameter describes the minimum number of neighbors for scoring each item.
        Since the LensKit default value for the min_nbrs hyperparameter is 1, we set the lower bound  to 1.
        The upper bound is set to the nnbrs hyperparameter value.
        Therefore, the upper bound of min_nbrs is set to 10000 to cover the full possible range of the
        min_nbrs hyperparameter.
        """
        min_nbrs = Integer('min_nbrs', bounds=(1, 10000), default=1, log=True)
        # FIXME: No good solution found for min_sim hyperparameter
        min_sim = Float('min_sim', bounds=(0, 0.1), default=0, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([min_nbrs, min_sim, nnbrs])

        return cs
