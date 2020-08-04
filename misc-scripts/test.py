
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(41)

N = 100
x = np.random.randint(0, 9, N)
bins = np.arange(10)

kde = stats.gaussian_kde(x)
xx = np.linspace(0, 9, 1000)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(x, density=False, bins=bins, alpha=0.3)
ax.plot(xx, kde(xx))
plt.show()
