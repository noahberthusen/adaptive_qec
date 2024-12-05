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
folder = "embedded"
r = 100

fig, ax = plt.subplots(1, 1, figsize=(4,3))

# K = 4
codes = [
    "HGP_C422_68_4",
    # "HGP_C422_104_4",
    # "HGP_C422_260_4",
    # "HGP_C422_3200_64",
]

codes3 = [
    "HGP_C422_68_4",
    # "HGP_C422_104_4",
    # "HGP_C422_260_4",
    # "HGP_C422_3200_64",
]

codes2 = [
    "HGP_34_4",
    # "HGP_52_4",
    "HGP_130_4",
    # "HGP_244_4",
]

labels = [
    "[[68,4,6]]",
    # "[[104,4,8]]",
    # "[[260,4,12]]",
    # "[[3200,64,20]]"
]

labels2 = [
    "[[34,4,3]]",
    # "[[52,4,4]]",
    "[[130,4,6]]",
    # "[[244,4,8]]"
]


# K = 6
codes = [
    "HGP_C422_160_16",
    # "HGP_C422_416_16"
]

codes3 = [
    "HGP_C422_160_16",
    # "HGP_C422_416_16",
]

codes2 = [
    "HGP_80_16",
    "HGP_208_16",
    # "HGP_400_16",
    # "HGP_976_16"
]

labels = [
    "[[160,16,6]]",
    # "[[416,16,12]]"
]

labels2 = [
    "[[80,16,3]]",
    "[[208,16,6]]",
    # "[[400,16,8]]",
    # "[[976,16,12]]"
]

ax_twin = ax.twinx()
ax_twin.set_yscale('log')
colors = ['#7F3C8D','#11A579','#3969AC','#F2B701','#e73f1e','#80BA5A','#E68310','#008695','#CF1C90','#f97b72','#4b4b8f','#A5AA99']
colors = colors[1:]
markers = ['d','o','p','s']


for i, code in enumerate(codes):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/unmasking/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    tmp_df = df[(df['r'] == r)]

    if not tmp_df.empty:
        ax.errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'],
                    fmt=f'{markers[i]}-', markersize=4, elinewidth=1.5, c=colors[i], label=labels[i])
        ax_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--",  c=colors[i])
        ax.set_yscale('log')
        ax.set_xscale('log')


for i, code in enumerate(codes2):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    tmp_df = df[(df['r'] == r)]

    if not tmp_df.empty:
        ax.errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'],
                    fmt=f'{markers[i+len(codes)]}-', markersize=4,
                    elinewidth=1.5, c=colors[i+len(codes)], label=labels2[i],)
        ax_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--",  c=colors[i+len(codes)])

        ax.set_yscale('log')
        ax.set_xscale('log')


for i, code in enumerate(codes3):
    df = pd.read_csv(os.path.join(path, f'../../results/{folder}/nonadaptive/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

    tmp_df = df[(df['r'] == r)]
    if not tmp_df.empty:
        ax.errorbar(tmp_df['p_phys'], tmp_df['ler_per_round'], tmp_df['error_bars'],
                    fmt=f'{markers[i+len(codes)+len(codes2)]}-', markersize=4, elinewidth=1.5,
                    c=colors[i+len(codes)+len(codes2)], label=labels[i],)
        ax_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--",  c=colors[i+len(codes)+len(codes2)])

        ax.set_yscale('log')
        ax.set_xscale('log')




handles1, labels1 = ax.get_legend_handles_labels()
first_legend = plt.legend(handles1, labels1, loc='upper left', fontsize=8.5)
ax_twin.add_artist(first_legend)

ax.set_ylabel(r"Logical error rate per round, $\epsilon_L$")
ax.set_xlabel(r"Error rate, $p$")
ax.set_xlabel(r"Error rate, $p$")
ax.set_xlabel(r"Error rate, $p$")
ax_twin.set_ylabel("Average CNOT gates per round")

plt.savefig(os.path.join(path, "../memory_circuit_lacross_single.png"), dpi=600, bbox_inches="tight")
