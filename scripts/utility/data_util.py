"""
Data utility methods to make life easier.
"""
import os
from pathlib import Path

import numpy as np


def get_data(dataset, data_dir='data'):
    """
    Returns a train and test set from the desired dataset.
    """
    in_dir = Path(data_dir) / dataset / 'continuous'
    assert in_dir.exists()

    train = np.load(in_dir / 'train.npy').astype(np.float32)
    test = np.load(in_dir / 'test.npy').astype(np.float32)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, X_test, y_train, y_test
