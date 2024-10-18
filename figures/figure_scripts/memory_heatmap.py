import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.patheffects as path_effects
import os
import pandas as pd

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

def get_data(concat, adaptive, qcode):
    f_path = f"../../results/"
    f_name = f"{qcode}.res"

    df = pd.read_csv(os.path.join(path, f_path+f_name))
    df = df[(df['adapt'] == adaptive) & (df['concat'] == concat)]
    df['p_error'] = 1 - df['p_log']
    df['p_std_dev'] = np.sqrt(df['p_error'] * df['p_log'] / df['no_test'])

    lifetime_data = np.zeros(shape=(5,5), dtype=float)

    for ii, i in enumerate(range(5,10)):
        phys_error = round(i*0.0002, 6)
        for jj, j in enumerate(range(5,10)):
            meas_error = round(j*0.0002, 6)

            row = df[(df['p_phys'] == phys_error) & (df['p_meas'] == meas_error)]
            lifetime_data[ii][jj] = row.iloc[0]['p_error']

    return lifetime_data

def annotate_heatmap(ax, data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{round(data[i, j],2)}x"  # Custom annotation
            text = ax.text(j, i, text, ha='center', va='center', color='white', fontsize=13)
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

plt.rc('font', family='serif')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1

# qcode1 = "HGP_C422_200_4.qcode"
qcode1 = "HGP_C642_150_4.qcode"

qcode2 = "HGP_900_36.qcode"
qcode3 = "HGP_100_4.qcode"

concat0adaptive0 = get_data(0,0,qcode3)
concat1adaptive0 = get_data(1,0,qcode1)
concat1adaptive1 = get_data(0,1,qcode1)

vmin = min(concat0adaptive0.min(), concat1adaptive0.min(), concat1adaptive1.min())
vmax = max(concat0adaptive0.max(), concat1adaptive0.max(), concat1adaptive1.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

im1 = axes[0].imshow(concat0adaptive0, cmap='viridis_r', origin='lower', vmin=vmin, vmax=vmax)
im2 = axes[1].imshow(concat1adaptive0, cmap='viridis_r', origin='lower', vmin=vmin, vmax=vmax)
im3 = axes[2].imshow(concat1adaptive1, cmap='viridis_r', origin='lower', vmin=vmin, vmax=vmax)

# annotate_heatmap(axes[0], concat0adaptive0)
annotate_heatmap(axes[1], concat1adaptive0/concat0adaptive0)
annotate_heatmap(axes[2], concat1adaptive1/concat0adaptive0)

axes[0].set_title(qcode3 + ' concat0adaptive0')
axes[1].set_title(qcode1 + ' concat0adaptive0')
axes[2].set_title(qcode1 + ' concat0adaptive1')

# fig.tight_layout()
cbar = fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Logical error rate')

axes[0].set_xlabel('$p_{meas}$')
axes[1].set_xlabel('$p_{meas}$')
axes[2].set_xlabel('$p_{meas}$')

axes[0].set_ylabel('$p_{phys}$')


plt.savefig(os.path.join(path, "../heatmaps.png"), dpi=600, bbox_inches="tight")