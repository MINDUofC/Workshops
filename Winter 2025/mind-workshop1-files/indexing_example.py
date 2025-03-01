import numpy as np

arr = np.array([10, 20, 30, 40, 50])

#Accessing the first element
print("Integer Indexing First Element:", arr[0])

# Access the last element (index -1)
print("Integer Indexing Last Element:", arr[-1])

# Create a boolean mask where elements are greater than 25
mask = arr > 25
print("Boolean Indexing Elements Greater than 25:", arr[mask])