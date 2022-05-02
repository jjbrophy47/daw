import os
import sys
import time
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
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


def main(args):

    X_train, X_test, y_train, y_test = load_data(args.dataset, args.data_dir)
    print(f'train shape: {X_train.shape}')

    # train
    tree = daw.DecisionTreeClassifier(topd=args.topd, k=args.k, max_depth=args.max_depth,
        random_state=args.random_state).fit(X_train, y_train)

    forest = daw.RandomForestClassifier(n_estimators=args.n_estimators, topd=args.topd,
        k=args.k, max_depth=args.max_depth, random_state=args.random_state).fit(X_train, y_train)

    # predict
    auc_tree = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    auc_forest = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
    print(f'\n[Tree] AUC: {auc_tree:.3f}')
    print(f'[Forest] AUC: {auc_forest:.3f}')

    # apply
    leaves_tree = tree.apply(X_test[:5])
    leaves_forest = forest.apply(X_test[:5])
    print(f'\n[Tree] leaves:\n{leaves_tree}')
    print(f'\n[Forest] leaves:\n{leaves_forest}')

    # slack
    slack_tree = tree.slack(X_test[:5])
    slack_forest = forest.slack(X_test[:5])
    print(f'\n[Tree] slack:\n{slack_tree}')
    print(f'\n[Forest] slack:\n{slack_forest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--topd', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=1)
    args = parser.parse_args()
    main(args)
