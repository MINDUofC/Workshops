import numpy as np

random_array = np.random.rand(10)
reshaped_random_array = random_array.reshape(2, 5)
print("Reshaped Random Array:\n", reshaped_random_array)
print("Mean:", np.mean(reshaped_random_array))
print("Sum:", np.sum(reshaped_random_array))
print("Max:", np.max(reshaped_random_array))
print("Min:", np.min(reshaped_random_array))
