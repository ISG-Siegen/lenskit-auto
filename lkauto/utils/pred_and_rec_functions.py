from lenskit import Pipeline
from lenskit.data import ItemListCollection
from lenskit.batch import predict as lk_predict
from lenskit.batch import recommend as lk_recommend

from lkauto.ensemble.greedy_ensemble_selection import EnsembleSelection


def predict(model, test_split: ItemListCollection):
    """
    Makes predictions on the test split. Can be used with a Pipeline model, or EnsembleSelection model.

    Parameters
    ----------
    model
        The model used for the prediction
    test_split: ItemListCollection
        The test split on which to perform the prediction

    Returns
    -------

    """
    if isinstance(model, EnsembleSelection):
        return model.predict(x_data=test_split)
    elif isinstance(model, Pipeline):
        return lk_predict(model, test_split)
    else:
        raise TypeError("model for prediction must be an EnsembleSelection or Pipeline")


def recommend(model, test_split: ItemListCollection):
    """
    Makes recommendations on the test split.

    Parameters
    ----------
    model
        The model used for the recommendation
    test_split
        The test split on which to perform the recommendation

    Returns
    -------

    """
    return lk_recommend(model, test_split)
