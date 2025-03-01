import math
from scipy import stats
import numpy as np

# Sample data
data = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23])

# Calculate mean and variance
mean = np.mean(data)
variance = np.var(data)

#sd is the square root of the variance
Standard_deviation = math.sqrt(np.var(data))

print("Mean:", mean)
print("Variance:", variance)

# Generate a random sample from a normal distribution
normal_sample = stats.norm.rvs(loc=0, scale=1, size=10)
print("Random Sample from Normal Distribution:", normal_sample)



