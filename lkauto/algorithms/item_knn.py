from lenskit.algorithms import item_knn
from ConfigSpace import Integer, Float
from ConfigSpace import ConfigurationSpace


class ItemItem(item_knn.ItemItem):
    def __init__(self, nnbrs, **kwargs):
        super().__init__(nnbrs=nnbrs, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        """
               return default configurationspace
               Default configuration spaces for hyperparameters are defined here.
        """

        """
        The nnbrs hyperparameter is set to 10000 in LensKit-Auto. Generally speaking, the higher the nnbrs
        hyperparameter value, the better the performance. But the computational cost will increase
        exponentially by increasing the nnbrs hyperparameter value. 10000 is a reasonable value for nnbrs
        hyperparameter since it has relatively good performance and is still able
        to run in a reasonable amount of time.
        """
        nnbrs = Integer('nnbrs', bounds=(1, 10000), default=1000, log=True)  # No default value given by LensKit

        """
        The min_sim hyperparameter describes the minimum number of neighbors for scoring each item.
        Since the LensKit default value for the min_nbrs hyperparameter is 1, we set the lower bound  to 1.
        The upper bound is set to the nnbrs hyperparameter value.
        Therefore, the upper bound of min_nbrs is set to 10000 to cover the full possible range of the
        min_nbrs hyperparameter.
        """
        min_nbrs = Integer('min_nbrs', bounds=(1, 1000), default=1, log=True)

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
        min_sim = Float('min_sim', bounds=(1.0e-10, 1.0e-2), default=1.0e-6)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([min_nbrs, min_sim, nnbrs])

        return cs
