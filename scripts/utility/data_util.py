"""
Data utility methods to make life easier.
"""
from pathlib import Path

import numpy as np
from sklearn.utils import check_array, check_consistent_length
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split


def get_data(dataset, data_dir='data', test_size=0.2, shuffle=True, random_state=1):
    """
    Returns a train and test set from the desired dataset.
    """
    if dataset in ['iris', 'cal']:
        data = load_iris() if dataset == 'iris' else fetch_california_housing()
        X, y = data['data'], data['target']

        if dataset == 'iris':
            objective, loss_fn = 'multiclass', log_loss_wrapper(labels=np.unique(y))
        else:
            objective, loss_fn = 'regression', squared_loss

        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=test_size, shuffle=shuffle, random_state=random_state)
    else:
        in_dir = Path(data_dir) / dataset / 'continuous'
        assert in_dir.exists()

        train = np.load(in_dir / 'train.npy').astype(np.float32)
        test = np.load(in_dir / 'test.npy').astype(np.float32)

        X_train, y_train = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]

        objective = 'binary'
        loss_fn = log_loss_wrapper(labels=np.unique(y_train))


    data = {'objective': objective,
            'loss_fn': loss_fn,
            'train': (X_train, y_train),
            'test': (X_test, y_test)}
    return data


def squared_loss(y_true, y_pred):
    """
    Input
    ---
        y_true: 1-D array of true values, shape=(len(y_true)).
        y_pred: 1-D array of predicted values, shape=(len(y_true)).

    Return
    ---
        1-D array of squared losses, shape=(len(y_true)).
    """
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true)
    return np.square(y_true - y_pred)


def log_loss_wrapper(labels):
    """
    Wrapper for log_loss function with predefined labels.
    """
    def wrapper(y_true, y_pred):
        return log_loss(y_true, y_pred, labels=labels)
    return wrapper


def log_loss(y_true, y_pred, eps=1e-15, labels=None):
    """
    Input
    ---
        y_true: 1-D array of true values, shape=(len(y_true)).
        y_pred: 1-D array of predicted values, shape=(len(y_true)).

    Return
    ---
        1-D array of logistic losses, shape=(len(y_true)).
    """
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true)

    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)

    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError(
                "y_true contains only one label ({0}). Please "
                "provide the true labels explicitly through the "
                "labels argument.".format(lb.classes_[0])
            )
        else:
            raise ValueError(
                "The labels array needs to contain at least two "
                "labels for log_loss, "
                "got {0}.".format(lb.classes_)
            )

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError(
                "y_true and y_pred contain different number of "
                "classes {0}, {1}. Please provide the true "
                "labels explicitly through the labels argument. "
                "Classes found in "
                "y_true: {2}".format(
                    transformed_labels.shape[1], y_pred.shape[1], lb.classes_
                )
            )
        else:
            raise ValueError(
                "The number of classes in labels is different "
                "from that in y_pred. Classes found in "
                "labels: {0}".format(lb.classes_)
            )

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    return loss
