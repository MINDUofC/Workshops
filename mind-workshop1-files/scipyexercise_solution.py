import math
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')                             # Use backend tag if running into issues with matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm

# Given data
times = np.array([0, 6, 12, 18, 24])
temps = np.array([8, 14, 20, 16, 10])

# Generate new x values for a smooth curve
x_new = np.linspace(0, 24, num=100)         # Creates 100 points between 0 and 24
f = interp1d(times, temps, kind='cubic')    # Uses interpolation function directly
y_new = f(x_new)

# Plot original data points and the interpolated curve
plt.plot(times, temps, 'o', label="Original Data")          # 'o' shows the original points
plt.plot(x_new, y_new, '-', label="Cubic Interpolation")    # '-' gives a smooth line

plt.legend()
plt.show()

# Step 2: Generate 10 simulated temperature variations
simulated_temps = norm.rvs(loc=np.mean(temps), scale= math.sqrt(np.var(temps)), size=10)
print("Standard Deviation of Data: ", math.sqrt(np.var(temps)))

print("Mean of Data: ", np.mean(temps))

print("Simulated temperatures:\n", simulated_temps)

# Step 3: Compute simulated variance
variance_temp = np.var(simulated_temps)

print("Variance of simulated temperatures:", variance_temp)

