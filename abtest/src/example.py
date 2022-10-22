from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


a = 60  # successes
b = 40  # failures
r = beta.rvs(a, b, size=10000)
plt.hist(r, density=True, bins=50, histtype='stepfilled', alpha=0.8)
plt.title("Beta Posterior Distribution of Conversion Rate for Variant A")
plt.show()

a2 = 30  # successes
b2 = 50  # failures
r2 = beta.rvs(a2, b2, size=10000)
plt.hist(r2, density=True, bins=50, histtype='stepfilled', alpha=0.8)
plt.title("Beta Posterior Distribution of Conversion Rate for Variant B")
plt.show()

delta = r - r2
print("probability that A is worse than B", np.mean(delta < 0))
print("probability that A is better than B", np.mean(delta >0))
plt.hist(delta, density=True, bins=50, histtype='stepfilled', alpha=0.8)
plt.title("Beta Posterior Distribution of the Difference Between Conversion Rate A and B")
plt.show()
