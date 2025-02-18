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
plt.rcParams['axes.linewidth'] = 1.2
folder = "lacross"
r = 100

fig, ax = plt.subplots(2, 1, figsize=(4,7),)

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
    # "HGP_130_4",
    # "HGP_244_4",
    "HGP_452_4",

]

labels = [
    "IB-HGP [[68,4,6]]",
    # "[[104,4,8]]",
    # "[[260,4,12]]",
    # "[[3200,64,20]]"
]

labels2 = [
    "HGP [[34,4,3]]",
    # "[[52,4,4]]",
    # "HGP [[130,4,6]]",
    # "HGP [[244,4,8]]"
    "HGP [[452,4,10]]"

]


# K = 6
codes4 = [
    # "HGP_C422_160_16",
    "HGP_C422_416_16"
]

codes6 = [
    # "HGP_C422_160_16",
    "HGP_C422_416_16",
]

codes5 = [
    # "HGP_80_16",
    "HGP_208_16",
    # "HGP_400_16",
    "HGP_976_16"
]

labels4 = [
    # "IB-HGP [[160,16,6]]",
    "IB-HGP [[416,16,12]]"
]

labels5 = [
    # "HGP [[80,16,3]]",
    "HGP [[208,16,6]]",
    # "HGP [[400,16,8]]",
    "HGP [[976,16,12]]"
]

ax_twin1 = ax[0].twinx()
ax_twin2 = ax[1].twinx()

def plot(ax, ax_twin, codes, codes2, codes3, labels, labels2):
    ax_twin.set_yscale('log')
    colors = ['#7F3C8D','#11A579','#3969AC','#F2B701','#e73f1e','#80BA5A','#E68310','#008695','#CF1C90','#f97b72','#4b4b8f','#A5AA99']
    colors = colors[1:]
    markers = ['d','o','p','s']

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
                        c=colors[i+len(codes)+len(codes2)], label=labels[i][:6] + " (NA)" + labels[i][6:],)
            ax_twin.plot(tmp_df['p_phys'], tmp_df['num_CNOTs'], linestyle="--",  c=colors[i+len(codes)+len(codes2)])

            ax.set_yscale('log')
            ax.set_xscale('log')

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




ax[0].set_ylabel(r"Logical error rate per round, $\epsilon_L$")
ax[1].set_ylabel(r"Logical error rate per round, $\epsilon_L$")

# fig.supylabel(r"Logical error rate per round, $\epsilon_L$")
# fig.supxlabel(r"Error rate, $p$")

ax[1].set_xlabel(r"Error rate, $p$")
ax_twin1.set_ylabel("Average CNOT gates per round")
ax_twin2.set_ylabel("Average CNOT gates per round")

plot(ax[0], ax_twin1, codes, codes2, codes3, labels, labels2)
plot(ax[1], ax_twin2, codes4, codes5, codes6, labels4, labels5)

handles1, labels1 = ax[0].get_legend_handles_labels()
first_legend = ax[0].legend(handles1, labels1, loc='upper left', fontsize=8.5, framealpha=0.7)
first_legend.remove()
ax_twin1.add_artist(first_legend)

handles2, labels2 = ax[1].get_legend_handles_labels()
second_legend = ax[1].legend(handles2, labels2, loc='upper left', fontsize=8.5, framealpha=0.7)
second_legend.remove()
ax_twin2.add_artist(second_legend)

plot_labels = ['(a)', '(b)']
for a, label in zip(ax, plot_labels):
    a.text(0.98, 0.03, label, transform=a.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right')


plt.savefig(os.path.join(path, "../memory_circuit_lacross.png"), dpi=600, bbox_inches="tight")
