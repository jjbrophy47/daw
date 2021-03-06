import os
import sys
import time
import argparse
import resource
from pathlib import Path
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import daw
from utility import data_util
from utility import print_util

def load_data(dataset, data_dir):

    if dataset == 'iris':
        data = load_iris()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        indices = np.where(y != 2)[0]
        X = X[indices]
        y = y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.2, shuffle=True, random_state=args.random_state)

    elif dataset == 'boston':
        data = load_boston()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        y = np.where(y < np.mean(y), 0, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.2, shuffle=True, random_state=args.random_state)

    else:
        X_train, X_test, y_train, y_test = data_util.get_data(dataset, data_dir)

    return X_train, X_test, y_train, y_test

def main(args):

    # experiment timer
    begin = time.time()

    # create output directory
    out_dir = Path(args.out_dir) / args.dataset / args.manipulation / f'depth_{args.max_depth}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # create logger
    logger = print_util.get_logger(out_dir / 'log.txt')
    logger.info(f'\nScript arguments: {args}')
    logger.info(f'\nTimestamp: {datetime.now()}')

    X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, data_dir=args.data_dir)
    logger.info(f'\nData:')
    logger.info(f'X_train: {X_train.shape}')
    logger.info(f'y_train: {y_train.shape}')
    logger.info(f'X_test: {X_test.shape}')
    logger.info(f'y_test: {y_test.shape}')

    # train
    model = daw.DecisionTreeClassifier(topd=args.topd, k=args.k, max_depth=args.max_depth,
        random_state=args.random_state).fit(X_train, y_train)
    logger.info(f'\nModel:\n{model}')

    # predict
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    logger.info(f'\nPrediction:\n{pred}')

    # evaluate
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    logger.info(f'\nAccuracy: {acc:.3f}, AUC: {auc:.3f}')

    # structural slack
    structure_slack = model.structure_slack(X_test, manipulation=args.manipulation)
    structure_slack = np.where(structure_slack <= 0, np.inf, structure_slack)
    structure_slack = np.min(structure_slack, axis=1)
    logger.info(f'\nMinimum structural slack:\n{structure_slack}')

    fig, ax = plt.subplots()
    sns.histplot(structure_slack, stat='percent', ax=ax)
    ax.set_ylabel('% test instances')
    ax.set_xlabel(f'No. {args.manipulation}s until structural change')
    ax.set_title(f'{args.dataset.capitalize()} ({X_train.shape[0]:,}) max_depth: {args.max_depth}')
    plt.savefig(out_dir / 'hist.pdf', bbox_inches='tight')

    # leaf slack
    leaf_slack = model.leaf_slack(X_test)
    logger.info(f'\nLeaf slack:\n{leaf_slack}')

    # save model results
    result = {}
    result['scipt_args'] = vars(args)
    result['model_params'] = model.get_params()
    result['n_features'] = X_train.shape[1]
    result['n_train'] = X_train.shape[0]
    result['n_test'] = X_test.shape[0]
    result['acc'] = acc
    result['auc'] = auc
    result['structure_slack'] = structure_slack
    result['leaf_slack'] = leaf_slack
    result['experiment_time'] = time.time() - begin
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    logger.info(f'\nResults:\n{result}')
    logger.info(f'\nSaving results to {out_dir}...')
    np.save(out_dir / 'results.npy', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data2')
    parser.add_argument('--out_dir', type=str, default='output/experiments/robustness/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--manipulation', type=str, default='deletion')
    parser.add_argument('--random_state', type=int, default=1)

    # Model settings
    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--topd', type=int, default=0)
    parser.add_argument('--k', type=int, default=1000000000)

    args = parser.parse_args()
    main(args)
