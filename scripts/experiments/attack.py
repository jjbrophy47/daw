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
from scipy.stats import gaussian_kde
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

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
    out_dir = Path(args.out_dir) / args.dataset / args.model / args.manipulation / f'depth_{args.max_depth}'
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
    if args.model == 'dt':
        model = DecisionTreeClassifier(max_depth=args.max_depth,
            random_state=args.random_state).fit(X_train, y_train)
    elif args.model == 'rf':
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
            random_state=args.random_state).fit(X_train, y_train)
    elif args.model == 'lr':
        model = LogisticRegression(solver=args.solver, max_iter=args.max_iter, C=args.C,
            random_state=args.random_state).fit(X_train, y_train)
    elif args.model == 'lgb':
        model = LGBMClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
            random_state=args.random_state).fit(X_train, y_train)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    logger.info(f'\nModel:\n{model}')

    # predict
    pred = model.predict(X_test).astype(np.int32)
    proba = model.predict_proba(X_test)
    logger.info(f'\nPrediction:\n{pred}')

    # evaluate
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba[:, 1])
    logger.info(f'\nAccuracy: {acc:.3f}, AUC: {auc:.3f}')

    # get number of correctly predicted test instances
    total_test_correct_idxs = np.where(pred == y_test)[0]
    total_n_test_correct = len(total_test_correct_idxs)
    logger.info(f'{total_n_test_correct}/{len(y_test)} correctly predicted test instances')

    # sample subset of correctly predicted test instances
    rng = np.random.default_rng(args.random_state)
    n_sample = min(args.max_test_correct, total_n_test_correct)
    test_correct_idxs = rng.choice(total_test_correct_idxs, size=n_sample, replace=False)
    n_test_correct = len(test_correct_idxs)
    logger.info(f'{n_test_correct}/{total_n_test_correct} correctly predicted test instances sampled')

    # get encoding of sampled correctly predicted test instances
    if args.manipulation in ['deletion', 'swap']:
        train_encoding = model.leaf_path(X_train)  # shape=(n_train, n_leaves)
        test_encoding = model.leaf_path(X_test)  # shape=(n_test, n_leaves)

    # attack
    logger.info(f'\nAttacking ({args.manipulation}s)')
    res_list = []
    start = time.time()
    progress_times = []
    for i, idx in enumerate(test_correct_idxs):

        # display progress
        if i  == 0:
            pass
        elif i % 10 == 0:
            progress_time = time.time() - start
            progress_times.append(progress_time)
            avg_progress_time = np.mean(progress_times)
            cum_progress_time = np.sum(progress_times)
            est_total_time = avg_progress_time * (len(test_correct_idxs) / 10)
            est_time_rem = est_total_time - cum_progress_time

            logger.info(f'\n{i}/{n_test_correct}, '
                        f'time: {progress_time:.2f}s, '
                        f'avg. time: {avg_progress_time:.2f}s, '
                        f'est. time remaining: {est_time_rem:.2f}s')
            logger.info(f'partial results: {res_list}')
            start = time.time()

        # initial prediction
        pred_init = pred_temp = pred[idx]

        # training influences on test instance
        if args.manipulation in ['deletion', 'swap']:
            sim = np.dot(train_encoding, test_encoding[idx])  # shape=(n_train,)
            sgn = np.where(y_train == pred_init, 1.0, -1.0)  # shape=(n_train,)
            influence = sim * sgn  # shape=(n_train,)
            influence_idxs = np.argsort(influence)[::-1]  # shape=(n_train,)

        res = 0

        # add examples to training set
        if args.manipulation == 'addition':

            # get test instance w/ opposite label
            x_test = X_test[[idx]]
            y_test_flip = 1 if y_test[idx] == 0 else 0

            # add copies of test instance (w/ opposite label) until pred changes
            # using a binary search
            l = 0
            r = len(X_train)
            while l < r:

                # add half the search range to the training set
                n_add = (r + l) // 2
                X_add_temp = np.repeat(x_test, repeats=n_add, axis=0)  # shape=(n_add, n_features)
                y_add_temp = np.repeat(y_test_flip, repeats=n_add, axis=0) # shape=(n_add,)

                X_train_temp = np.concatenate((X_train, X_add_temp), axis=0)
                y_train_temp = np.concatenate((y_train, y_add_temp), axis=0)

                # train a new model and re-predict
                model_temp = clone(model).fit(X_train_temp, y_train_temp)
                pred_temp = model_temp.predict(x_test)[0]

                # adjust search range
                if pred_temp != pred_init:
                    r = n_add
                    res = n_add
                else:
                    l = n_add + 1
        
        # remove examples from training set
        # NOTE: binary search MAY not ALWAYS find the LOWER BOUND based on removing SIMILAR examples,
        #       need to remove TEST EXAMPLE from the training set...
        elif args.manipulation == 'deletion':

            # get test instance
            x_test = X_test[[idx]]

            # remove most positively influential examples until pred changes using binary search
            l = 0
            r = len(X_train)
            while l < r:
                n_remove = (r + l) // 2
                remove_idxs = influence_idxs[:n_remove]

                X_train_temp = np.delete(X_train, remove_idxs, axis=0)
                y_train_temp = np.delete(y_train, remove_idxs)

                # train new model and re-predict
                model_temp = clone(model).fit(X_train_temp, y_train_temp)
                pred_temp = model_temp.predict(x_test)[0]
                
                # adjust search range
                if pred_temp != pred_init:
                    r = n_remove
                    res = n_remove
                else:
                    l = n_remove + 1

        # swap the labels from training set
        elif args.manipulation == 'swap':

            # get test instance
            x_test = X_test[[idx]]

            # initial training set
            y_train_temp = y_train.copy()

            # flip label of most positively influential examples until pred changes using binary search
            # NOTE: binary search does not ALWAYS find the LOWER BOUND, sometimes flipping
            #       the first e.g., 12 examples will change the prediction but NOT 13, but then again on 23...
            l = 0
            r = len(X_train)
            while l < r:
                n_swap = (r + l) // 2
                swap_idxs = influence_idxs[:n_swap]

                y_train_temp[swap_idxs] = 1 - y_train_temp[swap_idxs]  # flip

                # train new model and re-predict
                model_temp = clone(model).fit(X_train, y_train_temp)
                pred_temp = model_temp.predict(x_test)[0]

                y_train_temp[swap_idxs] = 1 - y_train_temp[swap_idxs]  # flip back

                # adjust search range
                if pred_temp != pred_init:
                    r = n_swap
                    res = n_swap
                else:
                    l = n_swap + 1

        else:
            raise ValueError(f'Unknown manipulation: {args.manipulation}')

        res_list.append(res)
    
    # upper lower bound on additions needed to flip prediction
    logger.info(f'\nNo. instances added to flip prediction:\n{res_list}')

    # CDF of additions needed to flip prediction
    _, ax = plt.subplots()
    x = np.sort(res_list)
    y = np.arange(1, len(x) + 1) / len(x) * 100
    ax.plot(x, y, '-')
    ax.set_ylabel('% test instances')
    ax.set_xlabel(f'No. {args.manipulation}s until prediction change')
    ax.set_title(f'{args.dataset.capitalize()} ({n_test_correct} correct test)')
    plt.savefig(out_dir / 'hist.pdf', bbox_inches='tight')

    # scatter plot of additions needed to flip prediction vs. confidence
    fig, ax = plt.subplots()
    x = np.array(res_list)
    y = proba[test_correct_idxs, pred[test_correct_idxs]]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    im = ax.scatter(x, y, c=z, s=50)
    ax.set_ylabel('Confidence (%)')
    ax.set_xlabel(f'No. {args.manipulation}s until prediction change')
    ax.set_title(f'{args.dataset.capitalize()} ({n_test_correct} correct test)')
    fig.colorbar(im, ax=ax)
    plt.savefig(out_dir / 'scatter.pdf', bbox_inches='tight')

    # save model results
    result = {}
    result['scipt_args'] = vars(args)
    result['model_params'] = model.get_params()
    result['n_features'] = X_train.shape[1]
    result['n_train'] = X_train.shape[0]
    result['n_test'] = X_test.shape[0]
    result['acc'] = acc
    result['auc'] = auc
    result['n_test_correct'] = n_test_correct
    result['test_correct_confidences'] = proba[test_correct_idxs, pred[test_correct_idxs]]
    result['test_correct_manipulations'] = np.array(res_list)
    result['experiment_time'] = time.time() - begin
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    logger.info(f'\nResults:\n{result}')
    logger.info(f'\nSaving results to {out_dir}...')
    np.save(out_dir / 'results.npy', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output/experiments/attack/')

    # Experiment settings
    parser.add_argument('--model', type=str, default='dt')
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--manipulation', type=str, default='addition')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--max_test_correct', type=int, default=1000)

    # Model settings
    parser.add_argument('--n_estimators', type=int, default=100)  # rf
    parser.add_argument('--max_depth', type=int, default=5)  # rf/dt
    parser.add_argument('--topd', type=int, default=0)  # dt
    parser.add_argument('--k', type=int, default=1000000000)  # dt
    parser.add_argument('--solver', type=str, default='liblinear')  # lr
    parser.add_argument('--max_iter', type=int, default=100)  # lr
    parser.add_argument('--C', type=float, default=1.0)  # lr

    args = parser.parse_args()
    main(args)
