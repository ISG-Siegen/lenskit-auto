import unittest

import numpy as np
import pandas as pd

from lkauto.utils.validation_split import validation_split
from lenskit.data import from_interactions_df


class TestValidationSplit(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(np.array([[1, 1, 1],
                                         [1, 2, 2],
                                         [1, 3, 3],
                                         [2, 1, 1],
                                         [2, 2, 2],
                                         [2, 3, 3],
                                         [3, 1, 1],
                                         [3, 2, 2],
                                         [3, 3, 3],
                                         [4, 1, 1],
                                         [4, 2, 2],
                                         [4, 3, 3],
                                         [5, 1, 1],
                                         [5, 2, 2],
                                         [5, 3, 3]]), columns=["user", "item", "rating", ])
        self.ds = from_interactions_df(self.df)

    """
    def test_validationSplit_givenValidDataFrame_correctSplitTrainAndValidationDataframesReturnedExpected(self):
        val_fold_indices = validation_split(data=self.ds, frac=0.25, random_state=42)

        validation_train = self.df.loc[val_fold_indices[0]["train"], :]
        validation_test = self.df.loc[val_fold_indices[0]["validation"], :]

        self.assertTrue(validation_train.shape == (75, 3))
        self.assertTrue(validation_test.shape == (25, 3))
        """

    def test_validationSplit_givenUnknownStrategy(self):
        self.assertRaises(ValueError,
                          validation_split, strategy="unknown", data=self.ds, frac=0.25, random_state=42)

    def test_validationSplit_givenValidDataset_1Fold_UserBased(self):
        splits = validation_split(data=self.ds, strategy="user_based", frac=0.2, num_folds=1, random_state=42)

        fold = next(splits)
        test_sample_fold = fold.test
        train_sample_fold = fold.train

        self.assertTrue(test_sample_fold.to_df().shape[0] == 3)
        self.assertTrue(train_sample_fold.interaction_count == 12)

    def test_validationSplit_givenValidDataset_5Fold_UserBaser(self):
        splits = validation_split(data=self.ds, strategy="user_based", num_folds=5, random_state=42)

        fold = next(splits)
        test_sample_fold = fold.test
        train_sample_fold = fold.train

        # 5 users with 3 ratings -> 1 user per fold
        # SampleFrac(0.2) of the 3 ratings leads to 1 rating for test fold, rest goes back to train set
        self.assertTrue(test_sample_fold.to_df().shape[0] == 1)
        self.assertTrue(train_sample_fold.interaction_count == 14)

    def test_validationSplit_givenValidDataset_1Fold_RowBaser(self):
        splits = validation_split(data=self.ds, strategy="row_based", frac=0.2, num_folds=1, random_state=42)

        fold = next(splits)
        test_sample_fold = fold.test
        train_sample_fold = fold.train

        self.assertTrue(test_sample_fold.to_df().shape[0] == 3)
        self.assertTrue(train_sample_fold.interaction_count == 12)

    def test_validationSplit_givenValidDataset_3Fold_RowBased(self):
        splits = validation_split(data=self.ds, strategy="row_based", frac=0.2, num_folds=3, random_state=42)

        fold = next(splits)
        test_sample_fold = fold.test
        train_sample_fold = fold.train

        self.assertTrue(test_sample_fold.to_df().shape[0] == 5)
        self.assertTrue(train_sample_fold.interaction_count == 10)


if __name__ == '__main__':
    unittest.main()
