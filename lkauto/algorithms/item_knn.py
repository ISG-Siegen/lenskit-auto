from lenskit.algorithms import item_knn
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter


class ItemItem(item_knn.ItemItem):
    def __init__(self, nnbrs, **kwargs):
        super().__init__(nnbrs=nnbrs, **kwargs)

    @staticmethod
    def get_default_configspace_hyperparameters():
        """
               return default configuration spaces for hyperparameter
        """
        item_item_min_nbrs = UniformIntegerHyperparameter('item_item_min_nbrs', lower=1, upper=250, default_value=1)
        item_item_min_sim = UniformFloatHyperparameter('item_item_min_sim', lower=0, upper=0.11, default_value=1.0e-6)

        return [item_item_min_nbrs, item_item_min_sim]
