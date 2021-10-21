import unittest

import numpy as np
import pandas as pd

from lkauto.utils.validation_split import validation_split


class TestValidationSplit(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(np.ones((100, 3)), columns=["user", "item", "rating", ])

    def test_validationSplit_givenValidDataFrame_correctSplitTrainAndValidationDataframesReturnedExpected(self):
        val_fold_indices = validation_split(data=self.df, frac=0.25, random_state=42)

        validation_train = self.df.loc[val_fold_indices[0]["train"], :]
        validation_test = self.df.loc[val_fold_indices[0]["validation"], :]

        self.assertTrue(validation_train.shape == (75, 3))
        self.assertTrue(validation_test.shape == (25, 3))


if __name__ == '__main__':
    unittest.main()
