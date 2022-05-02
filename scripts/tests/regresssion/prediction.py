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
    print(args)

    print(f'\nDATA')

    X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, data_dir=args.data_dir, random_state=args.random_state)
    print(f'\ntrain shape: {X_train.shape}')

    # train
    print(f'\nTRAIN')

    print('\nbuilding tree with criterion: squared_error...')
    tree_mse = daw.DecisionTreeRegressor(topd=args.topd, k=args.k, max_depth=args.max_depth,
        criterion='squared_error', random_state=args.random_state).fit(X_train, y_train)

    print('building tree with criterion: absolute_error...')
    tree_mae = daw.DecisionTreeRegressor(topd=args.topd, k=args.k, max_depth=args.max_depth,
        criterion='absolute_error', random_state=args.random_state).fit(X_train, y_train)

    print('building forest with criterion: squared_error...')
    forest_mse = daw.RandomForestRegressor(n_estimators=args.n_estimators, topd=args.topd,
        criterion='squared_error', k=args.k, max_depth=args.max_depth,
        random_state=args.random_state).fit(X_train, y_train)

    print('building forest with criterion: absolute_error...')
    forest_mae = daw.RandomForestRegressor(n_estimators=args.n_estimators, topd=args.topd,
        criterion='absolute_error', k=args.k, max_depth=args.max_depth,
        random_state=args.random_state).fit(X_train, y_train)

    # predict
    pred_tree_mse = tree_mse.predict(X_test)
    pred_tree_mae = tree_mae.predict(X_test)
    pred_forest_mse = forest_mse.predict(X_test)
    pred_forest_mae = forest_mae.predict(X_test)

    # evaluate
    print(f'\nEVALUATION')

    mse_tree_mse = mean_squared_error(y_test, pred_tree_mse)
    mse_tree_mae = mean_squared_error(y_test, pred_tree_mae)
    mse_forest_mse = mean_squared_error(y_test, pred_forest_mse)
    mse_forest_mae = mean_squared_error(y_test, pred_forest_mae)

    mae_tree_mse = median_absolute_error(y_test, pred_tree_mse)
    mae_tree_mae = median_absolute_error(y_test, pred_tree_mae)
    mae_forest_mse = median_absolute_error(y_test, pred_forest_mse)
    mae_forest_mae = median_absolute_error(y_test, pred_forest_mae)

    # display

    print(f'\nMetric: MSE (mean squared error)')
    print(f'[Tree - criterion: squared error]: {mse_tree_mse:.3f}')
    print(f'[Tree - absolute error]: {mse_tree_mae:.3f}')
    print(f'[Forest - squared error (mean of means)]: {mse_forest_mse:.3f}')
    print(f'[Forest - absolute error (median of medians)]: {mse_forest_mae:.3f}')

    print(f'\nMetric: MAE (median absolute error)')
    print(f'[Tree - squared error]: {mae_tree_mse:.3f}')
    print(f'[Tree - absolute error]: {mae_tree_mae:.3f}')
    print(f'[Forest - squared error (mean of means)]: {mae_forest_mse:.3f}')
    print(f'[Forest - absolute error (median of medians)]: {mae_forest_mae:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='diabetes')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--topd', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=1)
    args = parser.parse_args()
    main(args)
