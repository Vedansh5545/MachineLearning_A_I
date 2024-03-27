import numpy as np
import random




# Generate a 5x5 array of random integers between 1 and 100
a = np.random.randint(1, 101, size=(5, 5))

# Print the original array
print(a)

# Set all even numbers in the array to 0
a[a % 2 == 0] = 0

# Print the modified array
print(a)

