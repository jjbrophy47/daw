"""
Enumerate all possible binary datasets with p attributes,
and test whether adding the test instance w/ opposite label
changes the prediction less than adding any of the other
possible instances.
"""
import os
import sys
import time
import argparse
import resource
from pathlib import Path
from datetime import datetime
from itertools import product

import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import daw
from utility import print_util

def main(args):

    # experiment timer
    begin = time.time()

    # create output directory
    out_dir = Path(args.out_dir) / f'depth_{args.max_depth}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # create logger
    logger = print_util.get_logger(out_dir / 'log.txt')
    logger.info(f'\nScript arguments: {args}')
    logger.info(f'\nTimestamp: {datetime.now()}')

    # enumerate all possible instances for p binary attributes and a binary label
    bin_arr = [[0, 1]] * (args.p+1)
    instances = [np.array(x) for x in product(*bin_arr)]
    logger.info(f'\nPossible instances: {instances}')

    # enumerate all possible datasets assuming 0-X copies of each possible instance
    instance_counts_list = product(*[list(range(args.num_copies)) for _ in range(len(instances))])

    # create target instance
    x_test = np.array([1] * (args.p), dtype=np.float32).reshape(1, -1)  # shape=(1, p)
    y_test = np.array([1], dtype=np.int32)  # shape=(1,)
    y_test_opp = np.array([0], dtype=np.int32)  #shape=(1,)

    # evaluate each dataset
    n_counterexamples = 0
    i = 0
    for instance_counts in instance_counts_list:
        i += 1
        instance_counts = list(instance_counts)

        # construct set
        instance_list = []
        for j, n in enumerate(instance_counts):
            instance_list.append(np.tile(instances[j], (n, 1)))

        X = np.concatenate(instance_list, axis=0).astype(np.float32)
        if len(X) == 0:  # no instances, skip
            logger.info(f'Datasets evaluated: {i}')
            continue

        X_train = X[:, :-1]
        y_train = X[:, -1]

        # train
        model = daw.DecisionTreeClassifier(topd=args.topd, k=args.k, max_depth=args.max_depth,
            random_state=args.random_state).fit(X_train, y_train)

        # initial prediction
        proba_init = model.predict_proba(x_test)[0, 1]
        pred_init = 1 if proba_init > 0.5 else 0
        if pred_init != y_test[0]:  # incorrect prediction, skip
            logger.info(f'Datasets evaluated: {i}')
            continue

        # add test instance with opposite label and train a new model
        opp_X_train = np.concatenate((X_train, x_test.reshape(1, -1)), axis=0)
        opp_y_train = np.concatenate((y_train, y_test_opp), axis=0)
        opp_model = daw.DecisionTreeClassifier(topd=args.topd, k=args.k, max_depth=args.max_depth,
            random_state=args.random_state).fit(opp_X_train, opp_y_train)
        proba_opp = opp_model.predict_proba(x_test)[0, 1]
        diff_opp = proba_init - proba_opp  # diff_opp should likely be >= 0
    
        # add each possible instance one at a time, train, and compare prediction on diff_opp
        for instance in instances:
            new_X_train = np.concatenate((X_train, instance[:-1].reshape(1, -1)), axis=0)
            new_y_train = np.concatenate((y_train, instance[-1:]), axis=0)
            new_model = daw.DecisionTreeClassifier(topd=args.topd, k=args.k, max_depth=args.max_depth,
                random_state=args.random_state).fit(new_X_train, new_y_train)
            proba_instance = new_model.predict_proba(x_test)[0, 1]
            diff_instance = proba_init - proba_instance

            # compare prediction on diff_opp
            if diff_instance > diff_opp:
                logger.info(f'\nCounterexample found: {instance_counts} {instance}')
                logger.info(f'\n->initial X_train:\n{X_train}\ny_train: {y_train}\nmodel:\n{model}')
                logger.info(f'\n->opposite X_train:\n{opp_X_train}\ny_train: {opp_y_train}\nmodel:\n{opp_model}')
                logger.info(f'\n->other X_train:\n{new_X_train}\ny_train: {new_y_train}\nmodel:\n{new_model}')
                logger.info(f'\n->inital: {proba_init:.2f}, opposite: {proba_opp:.2f}, other: {proba_instance:.2f}')
                n_counterexamples += 1
                # exit(0)
        
        logger.info(f'Datasets evaluated: {i}')

    # save model results
    result = {}
    result['scipt_args'] = vars(args)
    result['n_counterexamples'] = n_counterexamples
    result['experiment_time'] = time.time() - begin
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    logger.info(f'\nResults:\n{result}')
    logger.info(f'\nSaving results to {out_dir}...')
    np.save(out_dir / 'results.npy', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='output/experiments/counterexample/')

    # Experiment settings
    parser.add_argument('--p', type=int, default=2)  # no. attributes
    parser.add_argument('--num_copies', type=int, default=2)  # max. no. copies of each instance
    parser.add_argument('--random_state', type=int, default=1)

    # Model settings
    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--topd', type=int, default=0)
    parser.add_argument('--k', type=int, default=1000000000)

    args = parser.parse_args()
    main(args)
