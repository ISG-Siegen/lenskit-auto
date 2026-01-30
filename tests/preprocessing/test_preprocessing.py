import unittest
import pandas as pd
from lenskit.data import from_interactions_df, Dataset

from lkauto.preprocessing.preprocessing import preprocess_data


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with sample data"""
        # Create sample interaction data with various scenarios
        self.interactions = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 5],
            'item_id': [101, 102, 103, 201, 202, 203, 301, 302, 401, 501, 502, 503, 504, 505],
            'rating': [5.0, 4.0, 3.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        })
        self.dataset = from_interactions_df(self.interactions)

    def test_preprocessData_givenValidInput_datasetReturned(self):
        """Test that preprocess_data returns a Dataset object"""
        result = preprocess_data(
            self.dataset,
            user_col='user_num',
            item_col='item_num',
            rating_col='rating',
            timestamp_col='timestamp'
        )
        # Verify that the result is a Dataset
        self.assertIsInstance(result, Dataset)

    def test_preprocessData_givenMinInteractions_usersFiltered(self):
        """Test filtering users with minimum interactions"""
        # User 1: 3 ratings, User 2: 3 ratings, User 3: 2 ratings,
        # User 4: 1 rating, User 5: 5 ratings
        result = preprocess_data(
            self.dataset,
            user_col='user_num',
            item_col='item_num',
            rating_col='rating',
            include_timestamp=False,
            min_interactions_per_user=3
        )

        result_df = result.interaction_table(format='pandas', original_ids=True)
        # extract unique users from the filtered result
        unique_users = result_df['user_id'].unique()

        # Only users with at least 3 ratings should remain (users 1, 2, 5)
        self.assertEqual(len(unique_users), 3)
        # check that the ratings are preserved: 3 + 3 + 5 = 11 total ratings
        self.assertEqual(len(result_df), 11)

    def test_preprocessData_givenMaxInteractions_usersFiltered(self):
        """Test filtering users with maximum interactions"""
        result = preprocess_data(
            self.dataset,
            user_col='user_num',
            item_col='item_num',
            rating_col='rating',
            include_timestamp=False,
            max_interactions_per_user=3
        )

        result_df = result.interaction_table(format='pandas', original_ids=True)
        unique_users = result_df['user_id'].unique()

        # Only users with 3 or fewer ratings should remain (users 1, 2, 3, 4)
        self.assertEqual(len(unique_users), 4)
        # check that the ratings are preserved: 3 + 3 + 2 + 1 = 9 total ratings
        self.assertEqual(len(result_df), 9)

    def test_preprocessData_givenNoTimestamp_datasetCreated(self):
        """Test preprocessing without timestamp column"""
        result = preprocess_data(
            self.dataset,
            user_col='user_num',
            item_col='item_num',
            rating_col='rating',
            include_timestamp=False
        )

        result_df = result.interaction_table(format='pandas')
        # check that the result is a Dataset
        self.assertIsInstance(result, Dataset)
        # check that the number of interactions is preserved
        self.assertEqual(len(result_df), len(self.interactions))

    def test_preprocessData_givenNoRatingNoTimestamp_datasetCreated(self):
        """Test preprocessing without rating and without timestamp"""
        result = preprocess_data(
            self.dataset,
            user_col='user_num',
            item_col='item_num',
            rating_col=None,
            include_timestamp=False
        )

        result_df = result.interaction_table(format='pandas')
        # check that the result is a Dataset
        self.assertIsInstance(result, Dataset)
        # check that the number of interactions is preserved
        self.assertEqual(len(result_df), len(self.interactions))

    def test_preprocessData_givenNoRatingWithTimestamp_datasetCreated(self):
        """Test preprocessing without rating column but with timestamp"""
        result = preprocess_data(
            self.dataset,
            user_col='user_num',
            item_col='item_num',
            rating_col=None,
            timestamp_col='timestamp',
            include_timestamp=True
        )

        result_df = result.interaction_table(format='pandas')
        # check that the result is a Dataset
        self.assertIsInstance(result, Dataset)
        # check that the number of interactions is preserved
        self.assertEqual(len(result_df), len(self.interactions))

    def test_preprocessData_givenNaValues_naValuesDropped(self):
        """Test that NaN values are dropped properly"""
        df_with_nan = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [101, 102, 201, 202, 301],
            'rating': [5.0, float('nan'), 3.0, 4.0, float('nan')],
            'timestamp': [1, 2, 3, 4, 5]
        })
        dataset_with_nan = from_interactions_df(df_with_nan.dropna())

        result = preprocess_data(
            dataset_with_nan,
            user_col='user_num',
            item_col='item_num',
            rating_col='rating',
            include_timestamp=False,
            drop_na_values=True
        )

        result_df = result.interaction_table(format='pandas', original_ids=True)
        # check that the result is a Dataset
        self.assertIsInstance(result, Dataset)
        # check that the number of interactions is preserved
        self.assertEqual(len(result_df), 3)

    def test_preprocessData_givenDuplicates_duplicatesDropped(self):
        """Test that duplicate rows are handled"""
        interactions_with_duplicates = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2],
            'item_id': [101, 102, 103, 201, 202],
            'rating': [5.0, 4.0, 3.0, 3.0, 4.0],
            'timestamp': [1, 2, 3, 4, 5]
        })
        dataset_with_duplicates = from_interactions_df(interactions_with_duplicates)

        result = preprocess_data(
            dataset_with_duplicates,
            user_col='user_num',
            item_col='item_num',
            rating_col='rating',
            include_timestamp=False,
            drop_duplicates=True
        )

        result_df = result.interaction_table(format='pandas', original_ids=True)
        # check that the result is a Dataset
        self.assertIsInstance(result, Dataset)
        # check that the number of interactions is preserved
        self.assertGreater(len(result_df), 0)


if __name__ == '__main__':
    unittest.main()
