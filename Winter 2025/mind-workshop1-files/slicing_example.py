import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Slice from index 1 to 3 (exclusive of 3)
print("Index 1-3:", arr[1:3])

# Slice from index 2 to the end
print("Index 2-end:", arr[2:])

# Slice from the beginning to index 3
print("Index beginning-3:", arr[:3])

# Slice with a step of 2
print("Step of 2:", arr[::2])

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Slice rows starting from index 1 and columns from index 1
print("Row and column index starting from 1:\n", arr_2d[1:, 1:])
