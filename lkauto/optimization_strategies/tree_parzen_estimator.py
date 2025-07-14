import numpy as np
import pandas as pd
import time

from ConfigSpace import ConfigurationSpace, Configuration

from lenskit.data import Dataset, ItemListCollection

from lkauto.explicit.explicit_evaler import ExplicitEvaler
from lkauto.implicit.implicit_evaler import ImplicitEvaler
from lkauto.utils.get_default_configurations import get_default_configurations
from lkauto.utils.filer import Filer
from lkauto.utils.get_default_configuration_space import get_default_configuration_space

from hyperopt import fmin, tpe, space_eval

from typing import Tuple
import logging


def tree_parzen(cs: ConfigurationSpace,
                  train: Dataset,
                  user_feedback: str,
                  optimization_metric,
                  filer: Filer,
                  validation: ItemListCollection = None,
                  time_limit_in_sec: int = 3600,
                  num_evaluations: int = None,
                  split_folds: int = 1,
                  split_strategie: str = 'user_based',
                  split_frac: float = 0.25,
                  ensemble_size: int = 50,
                  minimize_error_metric_val: bool = True,
                  num_recommendations: int = 10,
                  random_state=42) -> Tuple[Configuration, pd.DataFrame]: