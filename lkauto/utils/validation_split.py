import pandas as pd
import numpy as np


def validation_split(data: pd.DataFrame, frac: float = 0.25, random_state=42):
    # indices
    indices = {"train": np.array([]), "test": np.array([])}
    # rng instance for shuffling
    rng = np.random.default_rng(random_state)

    # as usual group by users and then sample from each user
    for user, items in data.groupby("user").indices.items():
        # shuffle items from each user
        rng.shuffle(items)
        train = items
        # the validation data is a split of the training data
        validation = rng.choice(train, round(items.shape[0] * frac), replace=False)
        indices['train'] = np.append(indices['train'], train)
        indices['validation'] = np.append(indices['validation'], validation)

    validation_train = data.iloc[indices["train"], :]
    validation = data.iloc[indices['validation'], :]
    return validation_train, validation
