from hyperopt import hp
from lenskit.sklearn.svd import BiasedSVDScorer
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter


class BiasedSVD(BiasedSVDScorer):
    def __init__(self, features, feedback="explicit", **kwargs):
        super().__init__(features=features, **kwargs)
        self.feedback = feedback
        self.features = features  # Store features as an instance variable for testing

    @staticmethod
    def get_default_configspace(number_item: int, hyperopt = False, **kwargs):
        """
               return default (hyperopt) configurationspace
        """

        """
        The algorithm is based on the scikit TruncatedSVD algorithm and the hyperparameters are
        choosen based around the default values of LensKit. No further explaination could be distracted
        from scikit or the original paper.
        """
        if number_item < 10000:
            default_value = number_item-1
        else:
            default_value = 1000
        n_items = min(number_item, 10000)

        if hyperopt:
            cs = {
                "algo": "BiasedSVD",
                "features": hp.uniformint("features", 2, n_items),
                "damping": hp.uniform("damping", 0.0001, 1000),
            }

        else:
            features = UniformIntegerHyperparameter('features', lower=2, upper=n_items, default_value=default_value, log=True)
            damping = UniformFloatHyperparameter('damping', lower=0.0001, upper=1000, default_value=5, log=True)

            cs = ConfigurationSpace()
            cs.add([features, damping])

        return cs
