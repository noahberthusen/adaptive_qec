import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os
from matplotlib.ticker import AutoMinorLocator
from quantum_code import *

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)


# plt.rc('font', family='serif')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2
r = 100

fig, ax = plt.subplots(2, 3, figsize=(10,4.3), gridspec_kw={'hspace': 0.25, 'wspace': 0.25})

codes2 = [
    "HGP_C422_200_4",
    "HGP_C422_800_16",
    "HGP_C422_1800_36"
]

codes3 = [
    "HGP_100_4",
    "HGP_400_16",
    "HGP_900_36"
]


codes4 = [
    "HGP_C422_160_16",
    "HGP_C422_416_16",
    "HGP_C422_800_16"
]

codes5 = [
    "HGP_106_16",
    "HGP_208_16",
    "HGP_400_16",
    "HGP_976_16"
]


codes6 = [
    "HGP_13_1",
    "HGP_25_1",
    "HGP_41_1",
    "HGP_61_1"
]

labels2 = "IB-Expander"
labels3 = "Expander"
labels4 = "IB-La-cross"
labels5 = "La-cross"
labels6 = "Surface"

def get_params(codes, family):
    distances = []
    physical_qubits = []

    for code in codes:
        if ("C422" in code):
            params = code.split("_")
            qcode_fname = os.path.join(path, f"../../codes/qcodes/{family}/HGP_{int(params[2])//2}_{params[3]}/{code}.qcode")
        else:
            qcode_fname = os.path.join(path, f"../../codes/qcodes/{family}/{code}/{code}.qcode")
        qcode = read_qcode(qcode_fname)
        d = min(len(l) for l in qcode.Lx+qcode.Lz)
        distances.append(d)
        physical_qubits.append(qcode.n)

    return distances, physical_qubits



def plot(row, col, codes, family, label, p, c, m):
    colors = ['#7F3C8D','#11A579','#3969AC','#F2B701','#e73f1e','#80BA5A','#E68310','#008695','#CF1C90','#f97b72','#4b4b8f','#A5AA99']
    colors = colors[1:]
    markers = ['d','o','p','s']

    distances, physical_qubits = get_params(codes, family)
    ler_per_round = []
    error_bars = []

    for i, code in enumerate(codes):
        if "C422" in code:
            df = pd.read_csv(os.path.join(path, f'../../results/{family}/unmasking/{code}.qcode.res'))
        else:
            df = pd.read_csv(os.path.join(path, f'../../results/{family}/{code}.qcode.res'))
        df['p_error'] = 1 - df['p_log']
        df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])
        df['p_error'].replace(to_replace=0, value=1e-4, inplace=True)
        df['ler_per_round'] = 1 - (1 - df['p_error'])**(1/df['r'])
        df['error_bars'] = (1 - df['p_error'])**(1/df['r']-1) * df['p_std_dev'] / df['r']

        tmp_df = df[(df['r'] == r)  & (df['p_phys'] == p)]
        ler_per_round.append(tmp_df['ler_per_round'].iloc[0])
        error_bars.append(tmp_df['error_bars'].iloc[0])


    # ax[0][col].errorbar(distances, ler_per_round, error_bars,
    #             fmt=f'{markers[m]}', markersize=5,
    #             elinewidth=1.5, c=colors[c],)# label=labels2[i],)

    # def exp_fun(x, c, V):
    #     return c / (np.abs(V)**((x+1)/2))
    # def fun(x, c, V):
    #     return np.log(c) - V*((x+1)/2)

    # # ax.errorbar(distances[inds], params[:,i][inds], errors[:,i][inds], fmt="o", c='k', label=label, marker=marker)
    # popt, pcov = curve_fit(fun, distances, np.log(ler_per_round), p0=(0.001, 0.2), maxfev=1000,)
    #     # sigma=np.log(errors[:,i][inds]))
    # std_dev = np.sqrt(np.diag(pcov))
    # print(p, codes[0], np.exp(popt), np.sqrt(np.diag(pcov)))

    # xx = np.linspace(distances[0], distances[-1], 100)

    # yy  = exp_fun(xx, popt[0], np.exp(popt[1]))
    # yy1 = exp_fun(xx, popt[0] + std_dev[0], np.exp(popt[1] - std_dev[1]))
    # yy2 = exp_fun(xx, popt[0] - std_dev[0], np.exp(popt[1] + std_dev[1]))
    # ax[0][col].plot(xx, yy, c=colors[c], label=r"$\Lambda$ = "+str(round(np.exp(popt[1]), 2)))
    # ax[0][col].plot(xx, yy1, c=colors[c], linestyle='--', alpha=0.2)
    # ax[0][col].plot(xx, yy2, c=colors[c], linestyle='--', alpha=0.2)

    ax[row][col].errorbar(physical_qubits, ler_per_round, error_bars,
                fmt=f'{markers[m]}-', markersize=4,
                elinewidth=1.5, c=colors[c], label=label)

def plot_surface(row, col, ks, codes, family, label, p, c, m):
    colors = ['#7F3C8D','#11A579','#3969AC','#F2B701','#e73f1e','#80BA5A','#E68310','#008695','#CF1C90','#f97b72','#4b4b8f','#A5AA99']
    colors = colors[1:]
    markers = ['d','o','p','s']

    distances, physical_qubits = get_params(codes, family)
    ler_per_round = []
    error_bars = []

    for i, code in enumerate(codes):
        if "C422" in code:
            df = pd.read_csv(os.path.join(path, f'../../results/{family}/unmasking/{code}.qcode.res'))
        else:
            df = pd.read_csv(os.path.join(path, f'../../results/{family}/{code}.qcode.res'))
        df['k_p_log'] = df['p_log']**ks[i]
        df['k_p_error'] = 1 - df['k_p_log']
        df['k_p_std_dev'] = np.sqrt(df['k_p_error'] * df['k_p_log'] / df['num_test'])
        df['k_p_error'].replace(to_replace=0, value=1e-4, inplace=True)
        df['ler_per_round'] = 1 - (1 - df['k_p_error'])**(1/df['r'])
        df['error_bars'] = (1 - df['k_p_error'])**(1/df['r']-1) * df['k_p_std_dev'] / df['r']

        tmp_df = df[(df['r'] == r)  & (df['p_phys'] == p)]
        ler_per_round.append(tmp_df['ler_per_round'].iloc[0])
        error_bars.append(tmp_df['error_bars'].iloc[0])

    ax[row][col].errorbar([q*k for q,k in zip(physical_qubits, ks)], ler_per_round, error_bars,
                fmt=f'{markers[m]}-', markersize=4,
                elinewidth=1.5, c=colors[c], label=label)





plot_surface(0,0,[4,16,36], ["HGP_41_1","HGP_41_1","HGP_41_1"], "surface", labels6, p=0.0001, c=2, m=2)
plot_surface(0,1,[4,16,36], ["HGP_41_1","HGP_41_1","HGP_41_1"], "surface", "", p=0.0005, c=2, m=2)
plot_surface(0,2,[4,16,36], ["HGP_41_1","HGP_41_1","HGP_41_1"], "surface", "", p=0.001, c=2, m=2)

plot(0,0, codes3, "expander", labels3, p=0.0001, c=1, m=1)
plot(0,1, codes3, "expander", "", p=0.0005, c=1, m=1)
plot(0,2, codes3, "expander", "", p=0.001, c=1, m=1)

plot(0,0, codes2, "expander", labels2, p=0.0001, c=0, m=0)
plot(0,1, codes2, "expander", "", p=0.0005, c=0, m=0)
plot(0,2, codes2, "expander", "", p=0.001, c=0, m=0)




plot_surface(1,0,[16]*len(codes6), codes6, "surface", labels6, p=0.0001, c=2, m=2)
plot_surface(1,1,[16]*len(codes6), codes6, "surface", "", p=0.0005, c=2, m=2)
plot_surface(1,2,[16]*len(codes6), codes6, "surface", "", p=0.001, c=2, m=2)

plot(1,0, codes5, "lacross", labels5, p=0.0001, c=1, m=1)
plot(1,1, codes5, "lacross", "", p=0.0005, c=1, m=1)
plot(1,2, codes5, "lacross", "", p=0.001, c=1, m=1)

plot(1,0, codes4, "lacross", labels4, p=0.0001, c=0, m=0)
plot(1,1, codes4, "lacross", "", p=0.0005, c=0, m=0)
plot(1,2, codes4, "lacross", "", p=0.001, c=0, m=0)




ax[0][0].set_title(f"$p = 0.01\%$")
ax[0][1].set_title(f"$p = 0.05\%$")
ax[0][2].set_title(f"$p = 0.1\%$")


ax[0][0].set_yscale('log')
ax[1][0].set_yscale('log')

ax[0][1].set_yscale('log')
ax[1][1].set_yscale('log')

ax[0][2].set_yscale('log')
ax[1][2].set_yscale('log')

# ax[1][0].set_xscale('log')
# ax[1][1].set_xscale('log')
# ax[1][2].set_xscale('log')


fig.supylabel(r"Logical error rate per round, $\epsilon_L$")
fig.supxlabel(r"Physical qubits, $n$")

# ax[1][1].set_xlabel(r"Physical qubits, $n$")


lines_labels = [ax.get_legend_handles_labels() for ax in ax[1]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc=(0.65,0.135), fontsize=9)


lines_labels = [ax.get_legend_handles_labels() for ax in ax[0]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels, loc=(0.335,0.595), fontsize=9)
fig.legend(lines, labels, loc=(0.65,0.774), fontsize=9)


labels = ["(a)", "(b)", "(c)"]
for tax, label in zip(ax[0], labels):
    tax.text(0.03, 0.03, label, transform=tax.transAxes, fontsize=10, verticalalignment='bottom')
labels = ["(d)", "(e)", "(f)"]
for tax, label in zip(ax[1], labels):
    if label == "(f)":
        tax.text(0.18, 0.03, label, transform=tax.transAxes, fontsize=10, verticalalignment='bottom')
    else:
        tax.text(0.03, 0.03, label, transform=tax.transAxes, fontsize=10, verticalalignment='bottom')

# for i in range(len(ax)):
#     for j in range(len(ax[0])):
#         ax[i][j].grid()
#         ax[i][j].grid(which='minor', alpha=0.3)

plt.savefig(os.path.join(path, "../overhead.png"), dpi=600, bbox_inches="tight")
# plt.tight_layout()
# plt.rcParams['figure.dpi'] = 600
# plt.show()