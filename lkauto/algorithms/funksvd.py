from hyperopt import hp
from lenskit.funksvd import FunkSVDScorer
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter


class FunkSVD(FunkSVDScorer):
    def __init__(self, features, feedback="explicit", **kwargs):
        super().__init__(features=features, **kwargs)
        # store feature as an instance variable so we can acces it (for testing)
        self.feedback = feedback
        self.features = features

    @staticmethod
    def get_default_configspace(hyperopt = False, **kwargs):
        """
               return default (hyperopt) configurationspace
        """

        if hyperopt:
            cs = {
                "algo": "FunkSVD",
                "features": hp.uniformint("features", 2, 10000),
                "lrate": hp.uniform("lrate", 0.0001, 0.01),
                "reg": hp.uniform("reg", 0.001, 0.1),
                "damping": hp.uniform("damping", 0.01, 1000),
            }

        else:
            features = UniformIntegerHyperparameter('features', lower=2, upper=10000, default_value=1000, log=True)

            """
            The authors of the original FunkSVD paper (https://sifter.org/~simon/journal/20061211.html) stated:
            Lrate is the learning rate, a rather arbitrary number which I fortuitously set to 0.001 on day one
            and regretted it every time I tried anything else after that. Err is the residual error from the
            current prediction.
            But the original dataset just evaluated the performance on the netflix price dataset. Other datasets
            perform well on ranges around 0.001.
            Therefore, the pip install -e .loatHyperparameter('lrate', lower=0.0001, upper=0.01, default_value=0.001)

            """
            lrate = UniformFloatHyperparameter('lrate', lower=0.0001, upper=0.01, default_value=0.001)
            """
            The authors of the original FunkSVD paper (https://sifter.org/~simon/journal/20061211.html) stated:
            The point here is to try to cut down on over fitting, ultimately allowing us to use
            more features. Last I recall, Vincent liked K=0.02 or so, with well over 100 features (singular vector
            pairs--if you can still call them that).
            The default value of 0.02 is considered for the range. The range is set to a close range around the 0.02 value.
            The default value is taken from the LensKit Library.
            """""
            reg = UniformFloatHyperparameter('reg', lower=0.001, upper=0.1, default_value=0.015)
            damping = UniformFloatHyperparameter('damping', lower=0.01, upper=1000, default_value=5, log=True)

            cs = ConfigurationSpace()
            cs.add([features, lrate, reg, damping])

        return cs
