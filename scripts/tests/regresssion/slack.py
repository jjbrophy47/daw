import os
import sys
import time
import argparse

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
import daw
from utility import data_util

def load_data(dataset, data_dir, random_state=1):

    if dataset == 'diabetes':
        data = load_diabetes()
        X = data['data']
        y = data['target']

    else:
        X_train, X_test, y_train, y_test = data_util.get_data(dataset, data_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=random_state)

    return X_train, X_test, y_train, y_test


def main(args):

    X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, data_dir=args.data_dir, random_state=args.random_state)
    print(f'train shape: {X_train.shape}')

    # train
    tree = daw.DecisionTreeRegressor(topd=args.topd, k=args.k, max_depth=args.max_depth,
        criterion=args.criterion, random_state=args.random_state).fit(X_train, y_train)

    forest = daw.RandomForestRegressor(n_estimators=args.n_estimators, topd=args.topd, k=args.k,
        max_depth=args.max_depth, criterion=args.criterion, random_state=args.random_state).fit(X_train, y_train)

    print(tree)
    print(forest)

    # predict
    pred_tree = tree.predict(X_test)
    pred_forest = forest.predict(X_test)
    mae_tree = median_absolute_error(y_test, pred_tree)
    mae_forest = median_absolute_error(y_test, pred_forest)
    print(f'[Tree - absolute error]: {mae_tree:.3f}')
    print(f'[Forest - absolute error]: {mae_forest:.3f}')

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
    parser.add_argument('--dataset', type=str, default='diabetes')
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--topd', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--criterion', type=str, default='absolute_error')
    parser.add_argument('--random_state', type=int, default=1)
    args = parser.parse_args()
    main(args)
