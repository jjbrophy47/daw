"""
Removes training examples based influence values computed by `influence.py`,
and computes change in prediction on each test example.
"""
import os
import sys
import time
import argparse
import resource
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from utility import data_util
from utility import print_util
from utility import exp_util

def main(args):

    # experiment timer
    begin = time.time()

    # create method str
    method = args.method
    if args.method == 'aki':
        method = f'aki_{args.k}'

    # create output directory
    in_dir = Path(args.in_dir) / args.dataset / args.model / method / f'seed_{args.random_state}'
    out_dir = Path(args.out_dir) / args.dataset / args.model / method / f'seed_{args.random_state}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # create logger
    logger = print_util.get_logger(out_dir / 'log.txt')
    logger.info(f'\nScript arguments: {args}')
    logger.info(f'\nTimestamp: {datetime.now()}')

    data = data_util.get_data(dataset=args.dataset, data_dir=args.data_dir)
    objective, loss_fn = data['objective'], data['loss_fn']
    X_train, y_train = data['train']
    X_test, y_test = data['test']
    logger.info(f'\nData:')
    logger.info(f'X_train: {X_train.shape}')
    logger.info(f'y_train: {y_train.shape}')
    logger.info(f'X_test: {X_test.shape}')
    logger.info(f'y_test: {y_test.shape}')
    logger.info(f'Objective: {objective} Loss function: {loss_fn}')

    # get model
    if args.model == 'lr':
        if objective == 'regression':
            model = LinearRegression()
        else:
            model = LogisticRegression(solver=args.solver, max_iter=args.max_iter, C=args.C,
                random_state=args.random_state)
    elif args.model == 'dt':
        class_fn = DecisionTreeRegressor if objective == 'regression' else DecisionTreeClassifier
        model = class_fn(max_depth=args.max_depth, random_state=args.random_state)
    elif args.model == 'rf':
        class_fn = RandomForestRegressor if objective == 'regression' else RandomForestClassifier
        model = class_fn(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)
    elif args.model == 'lgb':
        class_fn = LGBMRegressor if objective == 'regression' else LGBMClassifier
        model = class_fn(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    logger.info(f'\nModel:\n{model}')

    # get test indices and influence values
    inf_fp = in_dir / 'results.npy'
    if not inf_fp.exists():
        logger.info('Influence values not found, exiting...')
        return

    inf_res = np.load(inf_fp, allow_pickle=True)[()]
    test_sample = inf_res['test_sample']  # test indices
    influence = inf_res['influence']  # influence values, shape=(n_train, n_test)
    assert influence.shape[0] == len(X_train)
    logger.info(f'\nTest shape: {test_sample.shape}, influence shape: {influence.shape}')

    # remove training examples from most POSITIVELY influential to most NEGATIVELY influential
    sorted_train_idxs = np.argsort(influence, axis=0)[::-1]

    # result container
    loss = np.zeros((len(test_sample), len(args.remove_fracs)), dtype=np.float32)  # shape=(n_test, n_remove_frac)

    for i, test_idx in enumerate(test_sample):  # for each test example
        logger.info(f'\n#{i}, Test {test_idx}, label: {y_test[test_idx]}')
        x_test = X_test[[test_idx]]

        for j, remove_frac in enumerate(args.remove_fracs):
            n_remove = int(remove_frac * influence.shape[0])
            remove_idxs = sorted_train_idxs[:, i][:n_remove]

            new_X_train = np.delete(X_train, remove_idxs, axis=0)
            new_y_train = np.delete(y_train, remove_idxs, axis=0)
            new_model = clone(model).fit(new_X_train, new_y_train)

            if objective == 'regression':
                new_pred = new_model.predict(x_test)
            else:
                new_pred = new_model.predict_proba(x_test)

            # save result
            loss[i, j] = loss_fn(y_test[[test_idx]], new_pred)
            logger.info(f'Remove %: {remove_frac * 100:.1f}, prediction: {new_pred}, '
                        f'loss: {loss[i, j]:.5f}')

    # save model results
    result = {}
    result['script_args'] = vars(args)
    result['model_params'] = model.get_params()
    result['n_features'] = X_train.shape[1]
    result['n_train'] = X_train.shape[0]
    result['n_test'] = X_test.shape[0]
    result['loss'] = loss  # shape=(n_test, len(remove_fracs))
    result['experiment_time'] = time.time() - begin
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    logger.info(f'\nResults:\n{result}')
    logger.info(f'\nSaving results to {out_dir}...')
    np.save(out_dir / 'results.npy', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--in_dir', type=str, default='output/experiments/influence/')
    parser.add_argument('--out_dir', type=str, default='output/experiments/remove/')

    # Experiment settings
    parser.add_argument('--model', type=str, default='dt')
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--method', type=str, default='loo')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--remove_fracs', type=float, nargs='+',
        default=[0.0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15])

    # method settings
    parser.add_argument('--k', type=int, default=1)

    # Model settings
    parser.add_argument('--solver', type=str, default='liblinear')  # lr
    parser.add_argument('--max_iter', type=int, default=100)  # lr
    parser.add_argument('--C', type=float, default=1.0)  # lr
    parser.add_argument('--max_depth', type=int, default=5)  # dt/rf/lgb
    parser.add_argument('--n_estimators', type=int, default=100)  # rf/lgb

    args = parser.parse_args()
    main(args)
