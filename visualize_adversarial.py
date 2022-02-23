import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import load_pickle

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Dejavu Sans"]
})

SMALLEST_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

parser = argparse.ArgumentParser()
parser.add_argument('--metric_path_base', type=str, default=None, help='path to .pkl file containing the Metrics type object (from vanilla model)')
parser.add_argument('--metric_path_rand', type=str, default=None, help='path to .pkl file containing the Metrics type object (from random-trained model)')
parser.add_argument('--figure_path', type=str, default=None, help='path to save the figure to')
parser.add_argument('--criterion', type=str, default='ND', help='criterion of interest: should be one of MSE, MAPE, or ND')
parser.add_argument('--sigma_idx', type=int, default=-1, help='index of the column corresponding to smoothed model to plot')
parser.add_argument('--max_tolerance_idx', type=int, default=-1, help='index of the maximum index within the tolerance list to plot')

args = parser.parse_args()

metrics_base = list(load_pickle(args.metric_path_base))[0]
metrics_rand = list(load_pickle(args.metric_path_rand))[0]

criterion = args.criterion
sigma_idx = args.sigma_idx
max_idx = args.max_tolerance_idx

tolerance = [0.0] + metrics_base.tolerance
nd_table_base = metrics_base.to_table(criterion)
nd_table_rand = metrics_rand.to_table(criterion)

plt.plot(tolerance[:max_idx], np.maximum.accumulate(nd_table_base[:max_idx, 0]), label='Vanilla', color='mediumseagreen', marker='.', linewidth=1)
plt.plot(tolerance[:max_idx], np.maximum.accumulate(nd_table_rand[:max_idx, 0]), label='RT', color='indianred', marker='.', linewidth=1)
plt.plot(tolerance[:max_idx], np.maximum.accumulate(nd_table_base[:max_idx, sigma_idx]), label='RS', color='cornflowerblue', linestyle='-.', marker='.', linewidth=1)
plt.plot(tolerance[:max_idx], np.maximum.accumulate(nd_table_rand[:max_idx, sigma_idx]), label='RT+RS', color='mediumpurple', linestyle='-.', marker='.', linewidth=1)

plt.xlabel('Attack threshold')
plt.ylabel('Normalized deviation')

if args.figure_path is None:
    figure_path = 'figures/fig.png'
else:
    figure_path = args.figure_path

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(figure_path)
plt.show()
