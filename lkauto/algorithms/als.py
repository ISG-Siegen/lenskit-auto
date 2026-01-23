from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter


class ImplicitMF(ImplicitMFScorer):
    def __init__(self, embedding_size, feedback="implicit", **kwargs):
            if "damping_user" in kwargs and "damping_item" in kwargs:
                kwargs["damping"] = {
                    "user": kwargs.pop("damping_user"),
                    "item": kwargs.pop("damping_item")
                }

            super().__init__(
                feedback=feedback,
                embedding_size=embedding_size,
                **kwargs
            )
            self.embedding_size = embedding_size

    @staticmethod
    def get_default_configspace(**kwargs):
        """
              return default configurationspace
        """
        embedding_size = UniformIntegerHyperparameter(
            'embedding_size', lower=1, upper=10000, default_value=50, log=True
        )

        epochs = UniformIntegerHyperparameter(
            'epochs', lower=1, upper=100, default_value=10
        )

        regularization = UniformFloatHyperparameter(
            'regularization', lower=1e-4, upper=1.0, default_value=0.1, log=True
        )

        user_embeddings = CategoricalHyperparameter(
            'user_embeddings', choices=[True, False], default_value=True
        )

        # TEST-FORCED HACK
        damping_user = UniformFloatHyperparameter(
            'damping_user', lower=0.1, upper=20.0, default_value=5.0
        )

        damping_item = UniformFloatHyperparameter(
            'damping_item', lower=0.1, upper=20.0, default_value=5.0
        )

        cs = ConfigurationSpace()
        cs.add([
            embedding_size,
            regularization,
            epochs,
            damping_user,
            damping_item,
            user_embeddings
        ])
        return cs


class BiasedMF(BiasedMFScorer):
    def __init__(self, embedding_size, feedback="explicit", **kwargs):
            if "damping_user" in kwargs and "damping_item" in kwargs:
                kwargs["damping"] = {
                    "user": kwargs.pop("damping_user"),
                    "item": kwargs.pop("damping_item")
                }

            super().__init__(
                embedding_size=embedding_size,
                **kwargs
            )
            self.feedback = feedback
            self.embedding_size = embedding_size

    @staticmethod
    def get_default_configspace(**kwargs):
        """
               return default configuration spaces for hyperparameter
        """

        """
        The authors of the original ALS paper (https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32) stated:
        The most important discovery we made is that ALS-WR never overfits the data if we either increase
        the number of iterations or the number of hidden features.
        The paper stated that the improvement of the performance maximized around 1000 features.
        Therefore, we will set the upper bound and the default value of features to 10000.
        Since the authors just evaluated on one larger dataset, we still allow smaller and larger feature numbers
        but set the default value to 1000.
        """

        """
        The authors of the original ALS paper set the range of the regularization hyperparameter to from 0.03 - 0.065.
        https://link.springer.com/chapter/10.1007/978-3-540-68880-8_32
        Therefore we set the lower bound of the two regularization parameters (ureg and ireg) to 0.065.
        LensKit sets the default regularization hyperparameter to 0.1 Therefore we set the upper bound of the two
        regularization parameters (ureg and ireg) to 0.1.
        """
        embedding_size = UniformIntegerHyperparameter(
            'embedding_size', lower=1, upper=10000, default_value=50, log=True
        )

        epochs = UniformIntegerHyperparameter(
            'epochs', lower=1, upper=100, default_value=10
        )

        regularization = UniformFloatHyperparameter(
            'regularization', lower=1e-4, upper=1.0, default_value=0.1, log=True
        )

        user_embeddings = CategoricalHyperparameter(
            'user_embeddings', choices=[True, False], default_value=True
        )
        """
        The damping hyperparameter en- or disables a damping factor.
        In the future we may want to tune the damping values as well.
        """
        # TEST-FORCED HACK
        damping_user = UniformFloatHyperparameter(
            'damping_user', lower=0.1, upper=20.0, default_value=5.0
        )

        damping_item = UniformFloatHyperparameter(
            'damping_item', lower=0.1, upper=20.0, default_value=5.0
        )

        cs = ConfigurationSpace()
        cs.add([
            embedding_size,
            regularization,
            epochs,
            damping_user,
            damping_item,
            user_embeddings
        ])
        return cs
