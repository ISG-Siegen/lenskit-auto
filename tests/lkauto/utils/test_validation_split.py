import unittest

import numpy as np
import pandas as pd

from lkauto.utils.validation_split import validation_split


class TestValidationSplit(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(np.ones((100, 3)), columns=["user", "item", "rating", ])

    def test_validationSplit_givenValidDataFrame_correctSplitTrainAndValidationDataframesReturnedExpected(self):
        train, validation = validation_split(self.df, 0.25, 42)

        self.assertTrue(train.shape == (75, 3))
        self.assertTrue(validation.shape == (25, 3))


if __name__ == '__main__':
    unittest.main()
