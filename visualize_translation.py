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


figure_path = "./figures/translation_traffic_nips.pdf"

parser = argparse.ArgumentParser()
parser.add_argument('--table_path_base', type=str, default=None, help='path to .npy file containing the evaluation results for vanilla model')
parser.add_argument('--table_path_rand', type=str, default=None, help='path to .npy file containing the evaluation results for random-trained model')
parser.add_argument('--figure_path', type=str, default=None, help='path to save the figure to')
parser.add_argument('--sigma_idx', type=int, default=-1, help='index of the column corresponding to smoothed model to plot')

args = parser.parse_args()

table_base = np.load(args.table_path_base)
table_rand = np.load(args.table_path_rand)
sigma_idx = args.sigma_idx

perturbation_levels = np.array([-0.9, -0.8, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 9.0])

plt.plot(np.log10(1 + perturbation_levels), table_base[:, 0], label='vanilla', color='mediumseagreen', marker='.', linewidth=1)
plt.plot(np.log10(1 + perturbation_levels), table_rand[:, 0], label='rand-trained', color='indianred', marker='.', linewidth=1)
plt.plot(np.log10(1 + perturbation_levels), table_base[:, sigma_idx], label='smoothed', color='cornflowerblue', linestyle='-.', marker='.', linewidth=1)
plt.plot(np.log10(1 + perturbation_levels), table_rand[:, sigma_idx], label='rand-trained + smoothed', color='mediumpurple', linestyle='-.', marker='.', linewidth=1)

plt.xlabel('log(1+rho)')
plt.ylabel('Normalized deviation')

plt.legend()
plt.grid(True)
plt.savefig(args.figure_path)
plt.show()

