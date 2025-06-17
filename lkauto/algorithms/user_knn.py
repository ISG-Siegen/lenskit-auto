from lenskit.knn.user import UserKNNScorer, UserKNNConfig
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace import ConfigurationSpace


class UserUser(UserKNNScorer):
    def __init__(self, max_nbrs, min_nbrs=1, min_sim=1e-6, **kwargs):
        # store the parameters as an instance variables so we can acces it (for testing)
        self.max_nbrs = max_nbrs
        self.min_nbrs = min_nbrs
        self.min_sim = min_sim
        
        config= UserKNNConfig(
            max_nbrs=max_nbrs,
            min_nbrs=min_nbrs,
            min_sim=min_sim,
            **kwargs
        )
        super().__init__(config=config, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        """
            return default configuration spaces for hyperparameter
        """
        """
        The max_nbrs hyperparameter is set to 10000 in LensKit-Auto. Generally speaking, the higher the max_nbrs
        hyperparameter value, the better the performance. But the computational cost will increase
        exponentially by increasing the max_nbrs hyperparameter value. 10000 is a reasonable value for max_nbrs
        hyperparameter since it has relatively good performance and is still able
        to run in a reasonable amount of time.
        """
        max_nbrs = UniformIntegerHyperparameter('max_nbrs', lower=1, upper=10000, default_value=1000, log=True)

        """
        The min_nbrs hyperparameter describes the minimum number of neighbors for scoring each item.
        Since the LensKit default value for the min_nbrs hyperparameter is 1, we set the lower bound  to 1.
        The upper bound is set to the max_nbrs hyperparameter value.
        Therefore, the upper bound of min_nbrs is set to 10000 to cover the full possible range of the
        min_nbrs hyperparameter.
        """
        min_nbrs = UniformIntegerHyperparameter('min_nbrs', lower=1, upper=1000, default_value=1, log=True)

        """
        The min_sim hyperparameter describes the minimum threshold for similarity between items. It is commonly
        refered as the minimum support constraint. The min_sim hyperparameter limits the number of items that are taken
        into account for the similarity calculation.
        The following constrains are taken from :cite:t:`Deshpande2004-ht`
        A high value will result in a higher-order scheme that uses
        very few itemsets and as such it does not utilize its full potential, whereas a low value may lead to an
        exponentially large number of itemsets, making it computationally intractable.
        Unfortunately, there are no good ways to a priori select the value of support. This is because for a
        given value of σ the number of frequent item sets that exist in a dataset depends on the dataset’s density
        and the item co-occurrence patterns in the various rows.
        Since the paper already states that it is very difficult to find the best value, we define a large bound around
        the default LensKit value.
        """
        min_sim = UniformFloatHyperparameter('min_sim', lower=1.0e-10, upper=1.0e-2, default_value=1.0e-6, log=True)

        cs = ConfigurationSpace()
        cs.add([max_nbrs, min_nbrs, min_sim])

        return cs
