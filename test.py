import numpy as np

mu = 3.22
sigma = 0.45
theoretical_mean = np.exp(mu + (sigma ** 2) / 2)
theoretical_var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
theoretical_std = np.sqrt(theoretical_var)

print(theoretical_mean, theoretical_std)