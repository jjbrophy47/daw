import os
import sys
import time
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
import daw
from utility import data_util

def load_data(dataset, data_dir):

    if dataset == 'iris':
        data = load_iris()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        indices = np.where(y != 2)[0]
        X = X[indices]
        y = y[indices]

        X_train, X_test, y_train, y_test = X, X, y, y

    elif dataset == 'boston':
        data = load_boston()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        y = np.where(y < np.mean(y), 0, 1)

        X_train, X_test, y_train, y_test = X, X, y, y

    else:
        X_train, X_test, y_train, y_test = data_util.get_data(dataset, data_dir)

        X_train = X_train[:,:50]
        X_test = X_test[:,:50]

    return X_train, X_test, y_train, y_test


def sample_data_A():
    X_train = np.array([[0], [0], [1], [1]], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1], dtype=np.float32)
    X_test = np.array([[1]], dtype=np.float32)
    y_test = np.array([1], dtype=np.float32)
    return X_train, X_test, y_train, y_test

def sample_data_B():
    X_train = np.array([[0], [0], [1], [1]], dtype=np.float32)
    y_train = np.array([0, 1, 0, 1], dtype=np.float32)
    X_test = np.array([[1]], dtype=np.float32)
    y_test = np.array([1], dtype=np.float32)
    return X_train, X_test, y_train, y_test

def main(args):

    X_train, X_test, y_train, y_test = sample_data_B()
    print(f'X_train:\n{X_train}')
    print(f'y_train: {y_train}')
    print(f'X_test: {X_test}')
    print(f'y_test: {y_test}')

    # train
    tree = daw.DecisionTreeClassifier(topd=args.topd, k=args.k, max_depth=args.max_depth,
        random_state=args.random_state).fit(X_train, y_train)
    print(f'\ntree:\n{tree}')

    # predict
    pred = tree.predict_proba(X_test)
    print(f'\nprediction:\n{pred}')

    # apply
    leaf_ids = tree.apply(X_test)
    print(f'\nleaf IDs:\n{leaf_ids}')

    # slack
    leaf_slack = tree.leaf_slack(X_test)
    print(f'\nleaf slack:\n{leaf_slack}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--topd', type=int, default=0)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=1)
    args = parser.parse_args()
    main(args)
