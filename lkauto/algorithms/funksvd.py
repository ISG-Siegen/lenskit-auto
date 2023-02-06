from lenskit.algorithms import funksvd
from ConfigSpace import Integer, Float, ConfigurationSpace


class FunkSVD(funksvd.FunkSVD):
    def __init__(self, features, **kwargs):
        super().__init__(features=features, **kwargs)

    @staticmethod
    def get_default_configspace(**kwargs):
        """
               return default configurationspace
        """
        features = Integer('features', bounds=(2, 10000), default=1000, log=True)

        """
        The authors of the original FunkSVD paper (https://sifter.org/~simon/journal/20061211.html) stated:
        Lrate is the learning rate, a rather arbitrary number which I fortuitously set to 0.001 on day one
        and regretted it every time I tried anything else after that. Err is the residual error from the
        current prediction.
        But the original dataset just evaluated the performance on the netflix price dataset. Other datasets
        perform well on ranges around 0.001.
        Therefore, the default value is set to 0.0001 and the lower and upper bound are a close range around
        the default value.
        """
        lrate = Float('lrate', bounds=(0.0001, 0.01), default=0.001)

        """
        The authors of the original FunkSVD paper (https://sifter.org/~simon/journal/20061211.html) stated:
        The point here is to try to cut down on over fitting, ultimately allowing us to use
        more features. Last I recall, Vincent liked K=0.02 or so, with well over 100 features (singular vector
        pairs--if you can still call them that).
        The default value of 0.02 is considered for the range. The range is set to a close range around the 0.02 value.
        The default value is taken from the LensKit Library.
        """
        reg = Float('reg', bounds=(0.001, 0.1), default=0.015)
        damping = Float('damping', bounds=(0.01, 1000), default=5, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features, lrate, reg, damping])

        return cs
