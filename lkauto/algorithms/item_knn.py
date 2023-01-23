from lenskit.algorithms import item_knn
from ConfigSpace import Integer, Float
from ConfigSpace import ConfigurationSpace


class ItemItem(item_knn.ItemItem):
    def __init__(self, nnbrs, **kwargs):
        super().__init__(nnbrs=nnbrs, **kwargs)

    @staticmethod
    def get_default_configspace():
        """
               return default configurationspace
        """
        min_nbrs = Integer('min_nbrs', bounds=(1, 250), default=1)
        min_sim = Float('min_sim', bounds=(0, 0.11), default=1.0e-6)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([min_nbrs, min_sim])

        return cs
