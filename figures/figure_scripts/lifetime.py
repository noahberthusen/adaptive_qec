import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

qcode = "HGP_C422_400_8.qcode"
concat = 0
adaptive = 0
f_path = f"../../results/{qcode.strip('.qcode')}/concat{concat}adaptive{adaptive}/"


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


lifetime_data = np.zeros(shape=(5,5), dtype=float)

for ii, i in enumerate(range(10,15)):
    phys_error = round(i*0.0001, 6)
    for jj, j in enumerate(range(10,15)):
        meas_error = round(j*0.0001, 6)

        f_name = f"{qcode}_{concat}_{adaptive}_{phys_error}_{meas_error}.res"

        numbers_array = np.genfromtxt(f_path+f_name, delimiter=",")
        cleaned_array = numbers_array[~np.isnan(numbers_array)].astype(int)
        lifetime = expon_fit(cleaned_array)

        print(ii, jj, phys_error, meas_error, lifetime)
        lifetime_data[ii][jj] = lifetime[0]

plt.imshow(lifetime_data, cmap='viridis', interpolation='nearest', origin='lower')

# Customize the color bar
plt.colorbar(label='Color Scale')

# Add labels and title
plt.title(f'concat={concat},adaptive={adaptive}')
plt.xlabel('meas error')
plt.ylabel('phys error')

# Display the heatmap
plt.show()