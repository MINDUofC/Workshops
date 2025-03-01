import numpy as np
from scipy import interpolate
import matplotlib
matplotlib.use('TkAgg')                             # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt

# Known data points (e.g., measurements or observations)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1])

# Create an interpolation function based on known data points
f = interpolate.interp1d(x, y, kind='cubic')

# Generate new x values for a smooth curve by filling in the gaps
x_new = np.linspace(0, 5, 100)        # Creates 100 points between 0 and 5
y_new = f(x_new)                                      # Use the interpolation function to get y values for new x points

# Plot original data points and the interpolated curve
plt.plot(x, y, 'o', label="Original Data")  # 'o' shows the original points
plt.plot(x_new, y_new, '-', label="Cubic Interpolation")  # '-' gives a smooth line
plt.legend()
plt.show()


