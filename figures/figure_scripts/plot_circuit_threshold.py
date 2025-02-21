import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.patheffects as path_effects
import os
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib import ticker
from scipy.optimize import curve_fit

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)


plt.rcParams['axes.linewidth'] = 1

fig, ax = plt.subplots(2, 1, figsize=(4,4.5), sharex=True, sharey=True)

def threshold(ax, qcode, label, color, marker):
    f_path = f"../../results/"
    f_name = f"{qcode}.res"

    df = pd.read_csv(os.path.join(path, f_path+f_name))

    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    df = df[(df['p_std_dev'] > 0)]
    df = df[(df['p_phys'] < 0.0027)]


    df['p_error'].replace(to_replace=0, value=0.5, inplace=True)

    df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
    df['error_bars'] = (1 - df['p_error'])**(1/(df['r']-1)) * df['p_std_dev'] / df['r']

    ax.errorbar(df['p_phys'], df['ler_per_round'], df['error_bars'], label=label, fmt=f"{marker}-", c=color, ms=4)
    # ax.scatter(df['r'], df['ler_per_round'], label=qcode, marker='o', s=5)


colors = ['#7F3C8D','#11A579','#3969AC','#F2B701','#e73f1e','#80BA5A','#E68310','#008695','#CF1C90','#f97b72','#4b4b8f','#A5AA99']
colors = colors[1:]
markers = ['d','o','p','s']

# ler_per_round(ax[0], "cat10/HGP_100_4.qcode")
# ler_per_round(ax, "HGP_C422_200_4.qcode")
# ler_per_round(ax, "cats0/HGP_C422_200_4.qcode")
# ler_per_round(ax, "cats10/HGP_C422_200_4.qcode")
# folder = "FT0"
folder = "threshold"
folder2 = folder
folder3 = folder

threshold(ax[0], f"{folder3}/HGP_C422_200_4.qcode", "[[200,4,8]]", colors[0], markers[0])
threshold(ax[0], f"{folder3}/HGP_C422_800_16.qcode", "[[800,16,12]]", colors[1], markers[1])
threshold(ax[0], f"{folder3}/HGP_C422_1800_36.qcode", "[[1800,36,16]]", colors[2], markers[2])

# ler_per_round(ax[1], f"{folder2}/HGP_C422_200_4.qcode", "[[200,4,8]]", colors[0], markers[0])
# ler_per_round(ax[1], f"{folder2}/HGP_C422_800_16.qcode", "[[800,16,12]]", colors[1], markers[1])

threshold(ax[1], f"{folder}/HGP_100_4.qcode", "[[100,4,4]]", colors[0], markers[0])
threshold(ax[1], f"{folder}/HGP_400_16.qcode", "[[400,16,6]]",colors[1], markers[1])
threshold(ax[1], f"{folder}/HGP_900_36.qcode", "[[900,36,8]]",colors[2], markers[2])

ax[0].axvspan(0.0012, 0.0014, color='gray', alpha=0.3)
ax[1].axvspan(0.002, 0.0022, color='gray', alpha=0.3)

labels = ['(a)', '(b)', '(c)']
for a, label in zip(ax, labels):
    a.text(0.05, 0.95, label, transform=a.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left')

fig.text(-0.04, 0.5, r"Logical error rate per round, $\epsilon_L$", va='center', rotation='vertical')

ax[0].legend(loc='lower right', fontsize=9, framealpha=0.7)
ax[1].legend(loc='lower right', fontsize=9, framealpha=0.7)
ax[0].set_yscale('log')

ax[1].set_xlabel(r"Error rate, $p$")

# for i in range(len(ax)):
#     ax[i].grid()
#     ax[i].grid(which='minor', alpha=0.3)

plt.savefig(os.path.join(path, "../threshold.png"), dpi=600, bbox_inches="tight")
