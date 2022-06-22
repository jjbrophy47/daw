"""
Process robustness results.
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import spearmanr
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from utility import print_util


def main(args):

    # create output directory
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # create logger
    logger = print_util.get_logger(out_dir / 'log.txt')
    logger.info(f'\nScript arguments: {args}')
    logger.info(f'\nTimestamp: {datetime.now()}')

    # plot histograms for each manipulation/dataset/max_depth combination
    for manipulation in args.manipulation:
        logger.info(f'\n{manipulation}...')

        nrows, ncols = len(args.dataset), len(args.max_depth)
        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))

        for i, dataset in enumerate(args.dataset):
            logger.info(f'\n\t{dataset}...')

            for j, max_depth in enumerate(args.max_depth):
                logger.info(f'\t\tmax. depth: {max_depth}...')

                fp = Path(args.in_dir) / dataset / manipulation / f'depth_{max_depth}' / 'results.npy'
                if not fp.exists():
                    continue

                res = np.load(fp, allow_pickle=True)[()]

                # plot
                ax = axs[i][j]
                sns.histplot(res['structure_slack'], stat='percent', ax=ax)
                ax.set_ylabel('% test instances')
                ax.set_xlabel(f'# {manipulation}s until structural change')
                ax.set_title(f'max. depth: {max_depth}')
                plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

                if j == 0:
                    ax.set_ylabel(f'{dataset}\n'
                        f'{res["n_train"]:,}/{res["n_test"]:,} train/test\n'
                        f'% test instances')

        plt.tight_layout()
        plt.savefig(out_dir / f'{manipulation}.pdf', bbox_inches='tight')


    # plot leaf slack
    logger.info(f'\nleaf slack...')
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))

    for i, dataset in enumerate(args.dataset):
        logger.info(f'\n\t{dataset}...')

        for j, max_depth in enumerate(args.max_depth):
            logger.info(f'\t\tmax. depth: {max_depth}...')

            fp = Path(args.in_dir) / dataset / 'deletion' / f'depth_{max_depth}' / 'results.npy'
            if not fp.exists():
                continue

            res = np.load(fp, allow_pickle=True)[()]

            # plot
            ax = axs[i][j]
            sns.histplot(res['leaf_slack'], stat='percent', ax=ax)
            ax.set_ylabel('% test instances')
            ax.set_xlabel(f'# adds/dels until leaf pred. change')
            ax.set_title(f'max. depth: {max_depth}')
            plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

            if j == 0:
                ax.set_ylabel(f'{dataset}\n'
                    f'{res["n_train"]:,}/{res["n_test"]:,} train/test\n'
                    f'% test instances')

    plt.tight_layout()
    plt.savefig(out_dir / f'leaf.pdf', bbox_inches='tight')

    logger.info(f'\nSaved results to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/experiments/robustness/')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/robustness/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
        default=['adult', 'bank_marketing', 'census', 'credit_card', 'diabetes',
        'flight_delays', 'gas_sensor', 'higgs', 'no_show', 'olympics',
        'surgical', 'synthetic', 'twitter', 'vaccine'])
    # parser.add_argument('--dataset', type=str, nargs='+',
    #     default=['adult', 'bank_marketing', 'census', 'credit_card', 'diabetes',
    #     'flight_delays', 'gas_sensor', 'no_show', 'olympics',
    #     'surgical', 'twitter', 'vaccine'])
    parser.add_argument('--manipulation', type=str, nargs='+', default=['deletion', 'addition'])
    parser.add_argument('--max_depth', type=int, nargs='+', default=[1, 2, 3, 4, 5])

    args = parser.parse_args()
    main(args)
