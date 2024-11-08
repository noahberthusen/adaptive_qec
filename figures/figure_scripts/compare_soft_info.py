import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)


# plt.rc('font', family='serif')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1

fig, ax = plt.subplots(1, 3, figsize=(12,3), sharey=True)

codes = [
    "HGP_C422_200_4",
    "HGP_C422_800_16",
    "HGP_C422_1800_36",
    "HGP_C422_3200_64",
]

labels = [
    "[[200,4,8]]",
    "[[800,16,12]]",
    "[[1800,36,16]]",
    "[[3200,64,20]]"
]

for i, code in enumerate(codes):
    df = pd.read_csv(os.path.join(path, f'../../results/memory_phenom/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_std_dev'].replace(to_replace=0, value=1e-2, inplace=True)
    guesses = []
    params = []


    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # tmp_df = df[(df['p_std_dev'] > 0)]
    tmp_df = df[(df['soft'] == 1) & (df['adapt'] == 0)]
    ax[0].errorbar(tmp_df['p_phys'], tmp_df['p_error'], tmp_df['p_std_dev'], fmt='o', markersize=4, elinewidth=1.5,
                   label=labels[i])
    ax[0].set_title('Soft information')
    # ax[0].legend()

for i, code in enumerate(codes):
    df = pd.read_csv(os.path.join(path, f'../../results/memory_phenom/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_std_dev'].replace(to_replace=0, value=1e-2, inplace=True)
    guesses = []
    params = []


    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # tmp_df = df[(df['p_std_dev'] > 0)]
    tmp_df = df[(df['soft'] == 0) & (df['p_phys'] <= 0.02) & (df['adapt'] == 0)]
    ax[1].errorbar(tmp_df['p_phys'], tmp_df['p_error'], tmp_df['p_std_dev'], fmt='o', markersize=4, elinewidth=1.5,
                   label=labels[i])
    ax[1].set_title('Hard information')
    ax[1].legend()


for i, code in enumerate(codes):
    df = pd.read_csv(os.path.join(path, f'../../results/memory_phenom/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_std_dev'].replace(to_replace=0, value=1e-2, inplace=True)
    guesses = []
    params = []


    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # tmp_df = df[(df['p_std_dev'] > 0)]
    tmp_df = df[(df['soft'] == 1) & (df['p_phys'] <= 0.02) & (df['adapt'] == 1)]
    ax[2].errorbar(tmp_df['p_phys'], tmp_df['p_error'], tmp_df['p_std_dev'], fmt='o', markersize=4, elinewidth=1.5,
                   label=labels[i])
    ax[2].set_title('QED + QEC')
    # ax[2].legend()


ax[0].axvspan(0.02, 0.025, color='gray', alpha=0.3)
ax[1].axvspan(0.01, 0.015, color='gray', alpha=0.3)


ax[0].set_ylabel(r"Logical error rate, $p_{\log}$")
ax[1].set_xlabel(r"Error rate, $p$")
ax[0].set_xlabel(r"Error rate, $p$")
ax[2].set_xlabel(r"Error rate, $p$")


# plt.show()
plt.savefig(os.path.join(path, "../soft_compare.png"), dpi=600, bbox_inches="tight")
