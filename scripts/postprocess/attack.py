"""
Process robustness results.
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
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
        sharex = sharey = 'row' if args.share_xy else 'none'
        _, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), sharex=sharex, sharey=sharey)

        for i, dataset in enumerate(args.dataset):
            logger.info(f'\n\t{dataset}...')

            for j, max_depth in enumerate(args.max_depth):
                logger.info(f'\t\tmax. depth: {max_depth}...')

                fp = in_dir / dataset / manipulation / f'depth_{max_depth}' / 'results.npy'
                if not fp.exists():
                    continue

                res = np.load(fp, allow_pickle=True)[()]

                # plot
                ax = axs[i][j]
                sns.histplot(res['test_correct_manipulations'], stat='percent', ax=ax)
                ax.set_ylabel(f'% test ({res["n_test_correct"]} instances)')
                ax.set_xlabel(f'# {manipulation}s until prediction change')
                ax.set_title(f'max. depth: {max_depth}')
                plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

                axs[i][0].set_ylabel(f'{dataset}\n'
                    f'{res["n_train"]:,}/{res["n_test"]:,} train/test\n'
                    f'% test ({res["n_test_correct"]} instances)')

        plt.tight_layout()
        plt.savefig(out_dir / f'{manipulation}.pdf', bbox_inches='tight')

    logger.info(f'\nSaved results to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/talapas/attack/')
    parser.add_argument('--out_dir', type=str, default='output/postprocess/attack/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
        default=['adult', 'bank_marketing', 'census', 'credit_card', 'diabetes',
        'flight_delays', 'gas_sensor', 'higgs', 'no_show', 'olympics',
        'surgical', 'synthetic', 'twitter', 'vaccine'])
    parser.add_argument('--manipulation', type=str, nargs='+', default=['deletion', 'addition', 'swap'])
    parser.add_argument('--max_depth', type=int, nargs='+', default=[1, 2, 3, 4, 5])

    parser.add_argument('--share_xy', action='store_true')

    args = parser.parse_args()
    main(args)
