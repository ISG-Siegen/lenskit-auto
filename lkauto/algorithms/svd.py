from lenskit.sklearn.svd import BiasedSVDScorer
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter


class BiasedSVD(BiasedSVDScorer):
    def __init__(self, features, feedback="explicit", **kwargs):
        super().__init__(features=features, **kwargs)
        self.feedback = feedback
        self.features = features  # Store features as an instance variable for testing

    @staticmethod
    def get_default_configspace(number_item: int, **kwargs):
        """
               return default configurationspace
        """

        """
        The algorithm is based on the scikit TruncatedSVD algorithm and the hyperparameters are
        choosen based around the default values of LensKit. No further explaination could be distracted
        from scikit or the original paper.
        """
        n_items = number_item
        if n_items < 10000:
            features = UniformIntegerHyperparameter('features', lower=2, upper=n_items, default_value=n_items-1, log=True)
        else:
            # features = Integer('features', bounds=(2, 10000), default=1000, log=True)  # No default values given
            features = UniformIntegerHyperparameter('features', lower=2, upper=10000, default_value=1000, log=True)
        damping = UniformFloatHyperparameter('damping', lower=0.0001, upper=1000, default_value=5, log=True)

        cs = ConfigurationSpace()
        cs.add([features, damping])

        return cs
