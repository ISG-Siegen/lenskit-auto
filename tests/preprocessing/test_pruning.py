import unittest
import pandas as pd
from lenskit.data import from_interactions_df, Dataset

from lkauto.preprocessing.pruning import min_ratings_per_user, max_ratings_per_user


class TestPruning(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with sample data"""
        # Create sample interaction data
        # User 1 has 5 ratings (5 unique items)
        # User 2 has 3 ratings (3 unique items)
        # User 3 has 2 ratings (2 unique items)
        # User 4 has 1 rating (1 unique item)
        self.interactions = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
            'item_id': [101, 102, 103, 104, 105, 201, 202, 203, 301, 302, 401],
            'rating': [5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0]
        })
        self.dataset = from_interactions_df(self.interactions)

    def test_minRatingsPerUser_givenMinThreshold_usersWithFewerRatingsRemoved(self):
        """Test that users with fewer than min_ratings are removed"""
        result = min_ratings_per_user(self.dataset, num_ratings=3)

        # Only users with more than 3 ratings should remain (which are users 1, 2)
        result_df = result.interaction_table(format='pandas', original_ids=True)
        # extract unique users from the filtered result (should be 2 users)
        unique_users = result_df['user_id'].unique()

        # Check that only users with at least 3 ratings remain
        self.assertEqual(len(unique_users), 2)
        self.assertIn(1, unique_users)
        self.assertIn(2, unique_users)
        self.assertNotIn(3, unique_users)
        self.assertNotIn(4, unique_users)

    def test_minRatingsPerUser_givenMinThresholdOne_allUsersRemain(self):
        """Test that with min_ratings=1, all users remain"""
        result = min_ratings_per_user(self.dataset, num_ratings=1)

        # convert the result to a pandas DataFrame
        result_df = result.interaction_table(format='pandas', original_ids=True)
        # extract unique users from the filtered result (should be all 4 users)
        unique_users = result_df['user_id'].unique()
        # check that all 4 users remain
        self.assertEqual(len(unique_users), 4)

    def test_minRatingsPerUser_givenHighThreshold_noUsersRemain(self):
        """Test that with very high threshold, no users remain"""
        result = min_ratings_per_user(self.dataset, num_ratings=10)
        # convert the result to a pandas DataFrame
        result_df = result.interaction_table(format='pandas', original_ids=True)
        # check that no users remain
        self.assertEqual(len(result_df), 0)

    def test_maxRatingsPerUser_givenMaxThreshold_usersWithMoreRatingsRemoved(self):
        """Test that users with more than max_ratings are removed"""
        # Set max_ratings to 3 so users with more than 3 ratings are removed
        result = max_ratings_per_user(self.dataset, num_ratings=3)

        # Only users with 3 or fewer ratings should remain (users 2, 3, 4)
        result_df = result.interaction_table(format='pandas', original_ids=True)
        unique_users = result_df['user_id'].unique()

        # Check that only users with 3 or fewer ratings remain
        self.assertEqual(len(unique_users), 3)
        self.assertNotIn(1, unique_users)
        self.assertIn(2, unique_users)
        self.assertIn(3, unique_users)
        self.assertIn(4, unique_users)

    def test_maxRatingsPerUser_givenHighThreshold_allUsersRemain(self):
        """Test that with high max_ratings, all users remain"""
        result = max_ratings_per_user(self.dataset, num_ratings=10)
        # All users should remain since max_ratings is higher than all users' ratings
        result_df = result.interaction_table(format='pandas', original_ids=True)
        unique_users = result_df['user_id'].unique()
        # Check that all 4 users remain
        self.assertEqual(len(unique_users), 4)

    def test_maxRatingsPerUser_givenLowThreshold_onlyLowActivityUsersRemain(self):
        """Test that with low threshold, only users with few ratings remain"""
        result = max_ratings_per_user(self.dataset, num_ratings=1)

        # Only user with 1 rating should remain (user 4)
        result_df = result.interaction_table(format='pandas', original_ids=True)
        unique_users = result_df['user_id'].unique()
        # Check that only user 4 remains
        self.assertEqual(len(unique_users), 1)
        self.assertIn(4, unique_users)

    def test_minRatingsPerUser_returnsDatasetType(self):
        """Test that min_ratings_per_user function returns a Dataset object"""
        result = min_ratings_per_user(self.dataset, num_ratings=2)
        self.assertIsInstance(result, Dataset)

    def test_maxRatingsPerUser_returnsDatasetType(self):
        """Test that max_ratings_per_user function returns a Dataset object"""
        result = max_ratings_per_user(self.dataset, num_ratings=3)
        self.assertIsInstance(result, Dataset)
