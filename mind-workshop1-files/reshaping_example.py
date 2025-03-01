import numpy as np

# Creating a 1D array
arr = np.array([1, 2, 3, 4])

# Reshaping the 1D array into a 2D array with shape (2, 2)
reshaped_arr1 = arr.reshape(2, 2)
print("Reshaped Array:\n", reshaped_arr1)

# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5, 6])

# Let NumPy calculate the number of rows
reshaped_arr2 = arr.reshape(-1, 2)
print("Reshaped Array:\n", reshaped_arr2)
