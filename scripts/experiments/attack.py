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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
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

    # train_encoding = model.leaf_path(X_train)
    # test_encoding = model.leaf_path(X_test)
    # sim = np.dot(train_encoding, test_encoding.T)
    # sim0 = np.dot(train_encoding, test_encoding[0])
    # sim1 = np.dot(train_encoding, test_encoding[1])
    # print(sim, sim.shape)
    # print(sim0, sim0.shape)
    # assert np.all(sim0 == sim[:, 0])
    # assert np.all(sim1 == sim[:, 1])
    # sgn = np.where(y_train == y_test, 1.0, -1.0)

    # predict
    pred = model.predict(X_test)
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
        train_test_sim = np.dot(train_encoding, test_encoding.T)  # shape=(n_train, n_test)

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

        res = 0

        # add examples to training set
        if args.manipulation == 'addition':

            # get test instance w/ opposite label
            x_test_temp = X_test[[idx]]
            y_test_temp = [1] if y_test[idx] == 0 else [0]

            # initial training set
            X_train_temp = X_train.copy()
            y_train_temp = y_train.copy()

            # add copies of test instance (w/ opposite label) until pred changes
            while pred_temp == pred_init:
                X_train_temp = np.concatenate((X_train_temp, x_test_temp), axis=0)
                y_train_temp = np.concatenate((y_train_temp, y_test_temp), axis=0)

                # train new model and re-predict
                model_temp = clone(model).fit(X_train_temp, y_train_temp)
                pred_temp = model_temp.predict(x_test_temp)[0]
                res += 1
        
        # remove examples from training set
        elif args.manipulation == 'deletion':

            # get test instance
            x_test_temp = X_test[[idx]]

            # sort training examples by influence to the test instance using weighted leaf path
            sim = train_test_sim[:, idx]  # shape=(n_train,)
            sgn = np.where(y_train == pred_init, 1.0, -1.0)  # shape=(n_train,)
            influence = sim * sgn  # shape=(n_train,)
            influence_idxs = np.argsort(influence)[::-1]  # shape=(n_train,)

            # remove most positively influential examples until pred changes
            for i in range(len(influence_idxs)):
                X_train_temp = np.delete(X_train, influence_idxs[:i + 1], axis=0)
                y_train_temp = np.delete(y_train, influence_idxs[:i + 1])

                # train new model and re-predict
                model_temp = clone(model).fit(X_train_temp, y_train_temp)
                pred_temp = model_temp.predict(x_test_temp)[0]
                res += 1

                if pred_temp != pred_init:
                    break

        # swap the labels from training set
        elif args.manipulation == 'swap':

            # get test instance
            x_test_temp = X_test[[idx]]

            # initial training set
            y_train_temp = y_train.copy()

            # sort training examples by influence to the test instance using weighted leaf path
            sim = train_test_sim[:, idx]  # shape=(n_train,)
            sgn = np.where(y_train == pred_init, 1.0, -1.0)  # shape=(n_train,)
            influence = sim * sgn  # shape=(n_train,)
            influence_idxs = np.argsort(influence)[::-1]  # shape=(n_train,)

            # remove most positively influential examples until pred changes
            for influence_idx in influence_idxs:
                y_train_temp[influence_idx] = 1 if y_train_temp[influence_idx] == 0 else 0

                # train new model and re-predict
                model_temp = clone(model).fit(X_train, y_train_temp)
                pred_temp = model_temp.predict(x_test_temp)[0]
                res += 1

                if pred_temp != pred_init:
                    break

        else:
            raise ValueError(f'Unknown manipulation: {args.manipulation}')

        res_list.append(res)
    
    # upper lower bound on additions needed to flip prediction
    logger.info(f'\nNo. instances added to flip prediction:\n{res_list}')

    _, ax = plt.subplots()
    sns.histplot(res_list, stat='percent', ax=ax)
    ax.set_ylabel('% test instances')
    ax.set_xlabel(f'No. {args.manipulation}s until prediction change')
    ax.set_title(f'{args.dataset.capitalize()} ({n_test_correct} correct test) max_depth: {args.max_depth}')
    plt.savefig(out_dir / 'hist.pdf', bbox_inches='tight')

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
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--manipulation', type=str, default='addition')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--max_test_correct', type=int, default=100)

    # Model settings
    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--topd', type=int, default=0)
    parser.add_argument('--k', type=int, default=1000000000)

    args = parser.parse_args()
    main(args)
