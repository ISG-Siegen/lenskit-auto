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

        # Create data with NaN values
        self.interactions_with_nan = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [101, 102, 201, 202, 301],
            'rating': [5.0, None, 3.0, 4.0, None],
            'timestamp': [1, 2, 3, 4, 5]
        })
        self.dataset_with_nan = from_interactions_df(self.interactions_with_nan.dropna())

        # Create data with duplicates
        self.interactions_with_duplicates = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2],
            'item_id': [101, 102, 101, 201, 202],
            'rating': [5.0, 4.0, 5.0, 3.0, 4.0],
            'timestamp': [1, 2, 1, 4, 5]
        })
        self.dataset_with_duplicates = from_interactions_df(self.interactions_with_duplicates)

