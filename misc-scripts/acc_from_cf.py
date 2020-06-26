import numpy as np
cf = np.load("cf_target.npy")
print(np.trace(cf)/np.sum(cf))
