import unittest
from unittest.mock import MagicMock

from ConfigSpace import ConfigurationSpace, Categorical

from lkauto.optimization_strategies.random_search import random_search


class TestRandomSearch(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Create mock train dataset
        self.train = MagicMock()
        self.train.__class__.__name__ = 'Dataset'

        # Create mock validation
        self.validation = MagicMock()
        self.validation.__class__.__name__ = 'ItemListCollection'

        # Create a simple ConfigurationSpace
        self.cs = ConfigurationSpace(
            space={"algo": Categorical("algo", ["ItemItem", "UserUser"], default="ItemItem")}
        )

        # Create mock filer
        self.filer = MagicMock()

        # Create mock optimization metric
        self.optimization_metric = MagicMock()