import numpy as np

arr = np.array([1, 2, 3, 4])
scalar = 10

result1 = arr + scalar
print("Scalar and Array:", result1)

arr1 = np.array([1, 2, 3])              # Shape (3,)
arr2 = np.array([[10], [20], [30]])     # Shape (3, 1)

result2 = arr1 + arr2
print("Arrays of Different Sizes:\n", result2)