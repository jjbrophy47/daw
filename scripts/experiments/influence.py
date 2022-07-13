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
from influence import InfluenceEstimator

def main(args):

    # experiment timer
    begin = time.time()

    # create method str
    method = args.method
    if args.method == 'aki':
        method = f'aki_{args.k}'

    # create output directory
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

    if args.max_train is not None:
        X_train = X_train[:args.max_train].copy()
        y_train = y_train[:args.max_train].copy()
        logger.info(f'\nTrimmed training data:')
        logger.info(f'X_train: {X_train.shape}')
        logger.info(f'y_train: {y_train.shape}')

    # train
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
    model = model.fit(X_train, y_train)
    logger.info(f'\nModel:\n{model}')

    # evaluate
    logger.info('\nPredictive Performance:')
    eval_res = exp_util.eval_model(model, X_test, y_test, objective, loss_fn, logger)

    # select a subset of test instances to evaluate influence
    logger.info('\nInfluence:')
    rng = np.random.default_rng(args.random_state)

    # sample randomly from test set
    if objective == 'regression':
        test_sample = rng.choice(len(X_test), size=args.max_test_sample, replace=False)
    else:
        # get correclty predicted test instances
        test_correct_idxs = np.where(eval_res['pred'] == y_test)[0]
        logger.info(f'{len(test_correct_idxs)}/{len(y_test)} correctly predicted test instances')

        # sample randomly from correctly predicted test instances
        n_sample = min(args.max_test_sample, len(test_correct_idxs))
        test_sample = rng.choice(test_correct_idxs, size=n_sample, replace=False)
    
    logger.info(f'computing influence for {len(test_sample)} test instances...')

    # compute influence of training examples on test sample
    params = {}
    if args.method == 'aki':
        params = {'k': args.k}

    explainer = InfluenceEstimator(method=args.method, params=params, logger=logger)
    explainer.fit(model=model, X=X_train, y=y_train, objective=objective, loss_fn=loss_fn)
    influence = explainer.get_local_influence(X_test[test_sample], y_test[test_sample])  # (n_train, n_test_sample)
    logger.info(f'\nInfluence:\n{influence}, shape: {influence.shape}')

    # save model results
    result = {}
    result['script_args'] = vars(args)
    result['model_params'] = model.get_params()
    result['n_features'] = X_train.shape[1]
    result['n_train'] = X_train.shape[0]
    result['n_test'] = X_test.shape[0]
    result['n_test_sample'] = len(test_sample)
    result['test_sample'] = test_sample
    result['influence'] = influence
    result['experiment_time'] = time.time() - begin
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    logger.info(f'\nResults:\n{result}')
    logger.info(f'\nSaving results to {out_dir}...')
    np.save(out_dir / 'results.npy', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output/experiments/influence/')

    # Experiment settings
    parser.add_argument('--model', type=str, default='dt')
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--method', type=str, default='loo')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_test_sample', type=int, default=100)

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
