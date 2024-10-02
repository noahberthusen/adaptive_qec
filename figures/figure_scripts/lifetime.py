import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

def expon_fit(data):
    _, best_lifetime = sp.stats.expon.fit(data, floc=0)

    rng = np.random.default_rng(0)
    n_bs = 1000

    resampled_lifetimes = np.zeros(shape=n_bs)
    for i in range(n_bs):
        x_resamp = rng.choice(data, size=len(data))
        _, lifetime = sp.stats.expon.fit(x_resamp, floc=0)
        resampled_lifetimes[i] = lifetime

    lifetime_lower = np.quantile(resampled_lifetimes, 0.025)
    lifetime_upper = np.quantile(resampled_lifetimes, 0.975)

    return best_lifetime, (lifetime_lower, best_lifetime, lifetime_upper)

def get_data(concat, adaptive, qcode):
    f_path = f"../../results/{qcode.strip('.qcode')}/concat{concat}adaptive{adaptive}/"

    lifetime_data = np.zeros(shape=(5,5), dtype=float)

    for ii, i in enumerate(range(5,10)):
        phys_error = round(i*0.0002, 6)
        for jj, j in enumerate(range(5,10)):
            meas_error = round(j*0.0002, 6)

            f_name = f"{qcode}_{concat}_{adaptive}_{phys_error}_{meas_error}.res"

            numbers_array = np.genfromtxt(f_path+f_name, delimiter=",")
            cleaned_array = numbers_array[~np.isnan(numbers_array)].astype(int)
            lifetime = expon_fit(cleaned_array)

            print(phys_error, meas_error, lifetime[0])
            lifetime_data[ii][jj] = lifetime[0]
    return lifetime_data

def annotate_heatmap(ax, data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{round(data[i, j],2)}x"  # Custom annotation
            ax.text(j, i, text, ha='center', va='center', color='white')



qcode1 = "HGP_C422_400_8.qcode"
qcode2 = "HGP_400_16.qcode"
qcode3 = "HGP_225_9.qcode"

concat0adaptive0 = get_data(0,0,qcode1)
concat1adaptive0 = get_data(1,0,qcode1)
concat1adaptive1 = get_data(1,1,qcode1)

vmin = min(concat0adaptive0.min(), concat1adaptive0.min(), concat1adaptive1.min())
vmax = max(concat0adaptive0.max(), concat1adaptive0.max(), concat1adaptive1.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

im1 = axes[0].imshow(concat0adaptive0, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
im2 = axes[1].imshow(concat1adaptive0, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
im3 = axes[2].imshow(concat1adaptive1, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)

annotate_heatmap(axes[1], concat1adaptive0/concat0adaptive0)
annotate_heatmap(axes[2], concat1adaptive1/concat0adaptive0)

axes[0].set_title('concat0adaptive0')
axes[1].set_title('concat1adaptive0')
axes[2].set_title('concat1adaptive1')

# fig.tight_layout()
cbar = fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Average lifetime')

axes[0].set_xlabel('meas error')
axes[1].set_xlabel('meas error')
axes[2].set_xlabel('meas error')

axes[0].set_ylabel('phys error')


plt.savefig("../heatmaps.png", dpi=600)