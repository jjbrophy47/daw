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
from scipy.stats import gaussian_kde

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

    # get datasets
    datasets = [d for d in args.dataset if d not in args.ignore]

    # plot histograms for each manipulation/dataset/max_depth combination
    for manipulation in args.manipulation:
        logger.info(f'\n{manipulation}...')

        nrows, ncols = len(datasets), len(args.model)
        sharex = sharey = 'row' if args.share_xy else 'none'
        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), sharex=sharex, sharey=sharey)

        del_axes = []
        for i, dataset in enumerate(datasets):
            logger.info(f'\n\t{dataset}...')

            for j, model in enumerate(args.model):
                logger.info(f'\t\tmax. depth: {model}...')
                ax = axs[i][j]

                fp = in_dir / dataset / model / manipulation / f'depth_{args.depth}' / 'results.npy'
                if not fp.exists():
                    del_axes.append(ax)
                    continue

                # get results
                res = np.load(fp, allow_pickle=True)[()]

                # scatter
                if args.plot_type == 'scatter':
                    x = res['test_correct_manipulations']
                    y = res['test_correct_confidences'] * 100
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy)
                    im = ax.scatter(x, y, c=z, s=20)
                    ax.set_ylim(45, 105)
                    ax.set_ylabel(f'Confidence (%)')
                    axs[i][0].set_ylabel(f'{dataset}\n'
                        f'{res["n_train"]:,}/{res["n_test"]:,} train/test\n'
                        f'Confidence (%)')
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_ticklabels([])
                    cbar.set_ticks([])
                else:

                    # CDF
                    if args.plot_type == 'cdf':
                        x = np.sort(res['test_correct_manipulations'])
                        y = np.arange(1, len(x) + 1) / len(x) * 100
                        ax.plot(x, y, '-')

                    # histogram
                    else:
                        sns.histplot(res['test_correct_manipulations'], stat='percent', ax=ax)

                    ax.set_ylabel(f'% test ({res["n_test_correct"]} instances)')
                    axs[i][0].set_ylabel(f'{dataset}\n'
                        f'{res["n_train"]:,}/{res["n_test"]:,} train/test\n'
                        f'% test ({res["n_test_correct"]} instances)')

                ax.set_xlabel(f'# {manipulation}s til pred. change')
                ax.set_title(f'{model.upper()} ({res["auc"]:.3f})')
                plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

        for ax in del_axes:
            fig.delaxes(ax=ax)
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
    parser.add_argument('--ignore', type=str, nargs='+', default=['higgs', 'synthetic'])
    parser.add_argument('--model', type=int, nargs='+', default=['dt', 'lr', 'rf', 'lgb'])
    parser.add_argument('--manipulation', type=str, nargs='+', default=['addition'])
    parser.add_argument('--depth', type=int, default=5)

    # plot settings
    parser.add_argument('--plot_type', type=str, default='scatter')
    parser.add_argument('--share_xy', action='store_true')

    args = parser.parse_args()
    main(args)
