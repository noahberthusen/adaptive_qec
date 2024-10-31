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

fig, ax = plt.subplots(1, 2, figsize=(8,3), sharey=True)

codes = [
    "HGP_C422_200_4",
    "HGP_C422_800_16",
    "HGP_C422_1800_36",
    "HGP_C422_3200_64",
]

codes2 = [
    "HGP_100_4",
    "HGP_400_16",
    "HGP_900_36",
    "HGP_1600_64",
]

for i, code in enumerate(codes):
    df = pd.read_csv(os.path.join(path, f'../../results/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_std_dev'].replace(to_replace=0, value=1e-2, inplace=True)
    guesses = []
    params = []


    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # tmp_df = df[(df['p_std_dev'] > 0)]
    tmp_df = df
    ax[1].errorbar(tmp_df['p_phys'], tmp_df['p_error'], tmp_df['p_std_dev'], fmt='o', markersize=4, elinewidth=1.5,
                   label=f"{code[9:]}")
    # popt, pcov = curve_fit(fun, tmp_df['t'], tmp_df['p_error'], maxfev=1000, p0=(0.001), sigma=tmp_df['p_std_dev'])
    # print(code, popt, np.sqrt(pcov))
    # xx = np.linspace(2, 80, 1000)
    # yy = fun(xx, *popt)
    # ax[0].plot(xx, yy, c=colors[i], linewidth=1)
    ax[1].set_title('[[4,2,2]]-HGP codes')

    ax[1].legend()

for i, code in enumerate(codes2):
    df = pd.read_csv(os.path.join(path, f'../../results/{code}.qcode.res'))
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
    # df['p_std_dev'].replace(to_replace=0, value=1e-2, inplace=True)
    guesses = []
    params = []


    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # tmp_df = df[(df['p_std_dev'] > 0)]
    tmp_df = df
    ax[0].errorbar(tmp_df['p_phys'], tmp_df['p_error'], tmp_df['p_std_dev'], fmt='o', markersize=4, elinewidth=1.5,
                   label=f"{code[4:]}")
    # popt, pcov = curve_fit(fun, tmp_df['t'], tmp_df['p_error'], maxfev=1000, p0=(0.001), sigma=tmp_df['p_std_dev'])
    # print(code, popt, np.sqrt(pcov))
    # xx = np.linspace(2, 80, 1000)
    # yy = fun(xx, *popt)
    # ax[0].plot(xx, yy, c=colors[i], linewidth=1)
    ax[0].set_title('HGP codes')
    ax[0].legend()

# plt.show()
plt.savefig(os.path.join(path, "../memory_phenom.png"), dpi=600, bbox_inches="tight")
