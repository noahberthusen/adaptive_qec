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
folder = "lacross"
r = 100

fig, ax = plt.subplots(1, 1, figsize=(4,3))

# K = 4
# codes = [
#     "HGP_C422_68_4",
#     # "HGP_C422_104_4",
#     # "HGP_C422_260_4",
#     # "HGP_C422_3200_64",
# ]

# codes3 = [
#     "HGP_C422_68_4",
#     # "HGP_C422_104_4",
#     # "HGP_C422_260_4",
#     # "HGP_C422_3200_64",
# ]

# codes2 = [
#     "HGP_34_4",
#     # "HGP_52_4",
#     "HGP_130_4",
#     # "HGP_244_4",
# ]

# labels = [
#     "[[68,4,6]]",
#     # "[[104,4,8]]",
#     # "[[260,4,12]]",
#     # "[[3200,64,20]]"
# ]

# labels2 = [
#     "[[34,4,3]]",
#     # "[[52,4,4]]",
#     "[[130,4,6]]",
#     # "[[244,4,8]]"
# ]


# K = 6
codes = [
    # "HGP_C422_160_16",
    "HGP_C422_416_16"
]

codes3 = [
    "HGP_C422_416_16",
]

codes2 = [
    # "HGP_80_16",
    "HGP_208_16",
    "HGP_976_16"
]

labels = [
    # "[[160,16,6]]",
    "[[416,16,12]]"
]

labels2 = [
    # "[[80,16,3]]",
    "[[208,16,6]]",
    "[[976,16,12]]"
]

ax0_twin = ax.twinx()
ax0_twin.set_yscale('log')


for i, code in enumerate(codes3):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/nonadaptive/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    tmp_df = df[(df['r'] == r)]
    if not tmp_df.empty:
        ax.errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'], fmt='o-', markersize=4, elinewidth=1.5)
        ax0_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--", label=labels[i])

        ax.set_title('nonadaptive [[4,2,2]]-HGP codes')
        ax.set_yscale('log')
        ax.set_xscale('log')



for i, code in enumerate(codes2):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    tmp_df = df[(df['r'] == r)]

    if not tmp_df.empty:
        ax.errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'], fmt='o-', markersize=4, elinewidth=1.5)
        ax0_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--", label=labels2[i])

        ax.set_title('HGP codes')
        ax.set_yscale('log')
        ax.set_xscale('log')



for i, code in enumerate(codes):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/unmasking/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    tmp_df = df[(df['r'] == r)]

    if not tmp_df.empty:
        ax.errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'], fmt='o-', markersize=4, elinewidth=1.5)
        ax0_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--", label=labels[i])
        ax.set_title('[[4,2,2]]-HGP codes')
        ax.set_yscale('log')
        ax.set_xscale('log')


ax0_twin.legend(loc='upper left', frameon = True , facecolor = 'white', framealpha=1)

ax.set_ylabel(r"Logical error rate per round, $\epsilon_L$")
ax.set_xlabel(r"Error rate, $p$")
ax.set_xlabel(r"Error rate, $p$")
ax.set_xlabel(r"Error rate, $p$")
ax0_twin.set_ylabel("Average CNOT gates per round")

# plt.show()
plt.savefig(os.path.join(path, "../memory_circuit_lacross_single.png"), dpi=600, bbox_inches="tight")
