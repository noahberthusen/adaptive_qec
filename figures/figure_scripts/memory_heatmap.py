import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.patheffects as path_effects
import os
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib import ticker


full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

def get_data(concat, adaptive, qcode):
    f_path = f"../../results/heatmap/"
    f_name = f"{qcode}.res"

    df = pd.read_csv(os.path.join(path, f_path+f_name))
    df = df[(df['adapt'] == adaptive) & (df['concat'] == concat)]
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['num_test'])

    df['p_error'].replace(to_replace=0, value=0.5, inplace=True)


    lifetime_data = np.zeros(shape=(10,10), dtype=float)

    for ii, i in enumerate(range(1,11)):
        phys_error = round(i*0.0001, 6)
        for jj, j in enumerate(range(1,11)):
            meas_error = round(j*0.0001, 6)

            row = df[(df['p_phys'] == phys_error) & (df['p_meas'] == meas_error)]
            lifetime_data[ii][jj] = row.iloc[0]['p_error']

    return lifetime_data

def annotate_heatmap(ax, data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{round(data[i, j],2)}x"  # Custom annotation
            text = ax.text(j, i, text, ha='center', va='center', color='white', fontsize=8)
            # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

plt.rc('font', family='serif')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1

qcode1 = "HGP_C422_200_4.qcode"
# qcode1 = "HGP_C642_600_16.qcode"

# qcode2 = "HGP_C422_800_16.qcode"
qcode3 = "HGP_100_4.qcode"

concat0adaptive0 = get_data(0,0,qcode3)
concat1adaptive1 = get_data(1,1,qcode1)

vmin = min(concat0adaptive0.min(), concat1adaptive1.min())
vmax = max(concat0adaptive0.max(), concat1adaptive1.max())

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# im1 = axes[0].imshow(concat0adaptive0, cmap='viridis_r', origin='lower',  vmin=vmin, vmax=vmax,)
# im2 = axes[1].imshow(concat1adaptive0, cmap='viridis_r', origin='lower',  vmin=vmin, vmax=vmax,)
# im3 = axes[2].imshow(concat1adaptive1, cmap='viridis_r', origin='lower',  vmin=vmin, vmax=vmax,)
im1 = axes[0].imshow(concat0adaptive0, cmap='viridis_r', origin='lower',  norm=LogNorm(vmin=vmin, vmax=vmax,))
im3 = axes[1].imshow(concat1adaptive1, cmap='viridis_r', origin='lower',  norm=LogNorm(vmin=vmin, vmax=vmax,))

# annotate_heatmap(axes[0], concat0adaptive0)
# annotate_heatmap(axes[1], concat0adaptive0/concat1adaptive1)
annotate_heatmap(axes[1], concat1adaptive1/concat0adaptive0)

axes[0].set_title(qcode3)
axes[1].set_title(qcode1)

# for i in range(3):
#     axes[i].set_xticklabels(['0.001', '0.002', '0.003', '0.004', '0.005', '0.006'])

# positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# labels = ['0.001', '0.002', '0.003', '0.004', '0.005', '0.0006']
# for i in range(3):
    # axes[i].xaxis.set_major_locator(ticker.FixedLocator(positions))
    # axes[i].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
# axes[0].yaxis.set_major_locator(ticker.FixedLocator(positions))
# axes[0].yaxis.set_major_formatter(ticker.FixedFormatter(labels))

fig.tight_layout()
cbar = fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Logical error rate')

axes[0].set_xlabel('$p_{meas}$')
axes[1].set_xlabel('$p_{meas}$')
# axes[2].set_xlabel('$p_{meas}$')

axes[0].set_ylabel('$p_{phys}$')


plt.savefig(os.path.join(path, "../heatmaps.png"), dpi=600, bbox_inches="tight")