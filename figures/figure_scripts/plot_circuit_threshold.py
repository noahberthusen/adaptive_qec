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
folder = "memory_circuit"

fig, ax = plt.subplots(1, 3, figsize=(12,3), sharey=True)

codes = [
    "HGP_C422_200_4",
    "HGP_C422_800_16",
    # "HGP_C422_1800_36",
    # "HGP_C422_3200_64",
]

codes3 = [
    "HGP_C422_200_4",
    "HGP_C422_800_16",
    # "HGP_C422_1800_36",
    # "HGP_C422_3200_64",
]

codes2 = [
    "HGP_100_4",
    "HGP_400_16",
    "HGP_900_36",
    # "HGP_1600_64",
]

labels = [
    "[[200,4,8]]",
    "[[800,16,12]]",
    "[[1800,36,16]]",
    "[[3200,64,20]]"
]

labels2 = [
    "[[100,4,4]]",
    "[[400,16,6]]",
    "[[900,36,8]]",
    "[[1600,64,10]]"
]

ax2_twin = ax[2].twinx()
ax1_twin = ax[1].twinx()
ax0_twin = ax[0].twinx()
ax2_twin.set_yscale('log')
ax1_twin.set_yscale('log')
ax0_twin.set_yscale('log')

ax2_twin.get_shared_y_axes().join(ax0_twin, ax1_twin, ax2_twin)


for i, code in enumerate(codes3):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/nonadaptive/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # tmp_df = df[(df['adapt'] == 1)]
    tmp_df = df[(df['p_std_dev'] > 0)]

    ax[2].errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'], fmt='o-', markersize=4, elinewidth=1.5,
                   label=labels[i])
    ax2_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--")

    ax[2].set_title('nonadaptive [[4,2,2]]-HGP codes')
    ax[2].legend(loc='lower right')
    ax[2].set_yscale('log')
    ax[2].set_xscale('log')



for i, code in enumerate(codes2):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # tmp_df = df[(df['p_std_dev'] > 0)]
    tmp_df = df
    ax[1].errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'], fmt='o-', markersize=4, elinewidth=1.5,
                   label=labels2[i])
    ax1_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--")

    ax[1].set_title('HGP codes')
    ax[1].legend(loc='lower right', )
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')



for i, code in enumerate(codes):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/FT/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    tmp_df = df
    tmp_df = df[(df['p_std_dev'] > 0)]

    ax[0].errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'], fmt='o-', markersize=4, elinewidth=1.5,
                   label=labels[i])
    ax0_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--")
    ax[0].set_title('[[4,2,2]]-HGP codes')
    ax[0].legend(loc='upper left')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')




ax[0].set_ylabel(r"Logical error rate per round, $\epsilon_L$")
ax[2].set_xlabel(r"Error rate, $p$")
ax[1].set_xlabel(r"Error rate, $p$")
ax[0].set_xlabel(r"Error rate, $p$")
ax2_twin.set_ylabel("Average CNOT gates per round")

# plt.show()
plt.savefig(os.path.join(path, "../memory_circuit.png"), dpi=600, bbox_inches="tight")
