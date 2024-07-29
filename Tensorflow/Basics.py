'''
This code demonstrates the basics of TensorFlow. It covers the following topics:
- Initialization of Tensors
- Mathematical operations on Tensors
- Indexing and slicing Tensors
- Reshaping and transposing Tensors
'''

# Importing the required libraries

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

physical_devices = tf.config.list_physical_devices('GPU')

# Initialization of Tensors

x = tf.constant(4, shape=(1,1), dtype=tf.float32)  # Create a constant tensor x with shape (1,1) and value 4

y = tf.constant([[1,2,3], [4,5,6]], shape=(2,3), dtype=tf.int32)  # Create a constant tensor y with shape (2,3) and values [[1,2,3], [4,5,6]]

z = tf.ones((3,3))  # Create a tensor z with shape (3,3) and all elements set to 1

a = tf.zeros((2,3))  # Create a tensor a with shape (2,3) and all elements set to 0

b = tf.eye(3)  # Create a tensor b with shape (3,3) and diagonal elements set to 1 and others set to 0

c = tf.random.normal((3,3), mean=0, stddev=1)  # Create a tensor c with shape (3,3) and random values from a normal distribution with mean 0 and standard deviation 1

d = tf.random.uniform((1,3), minval=0, maxval=1)  # Create a tensor d with shape (1,3) and random values from a uniform distribution between 0 and 1

e = tf.range(1,10,2)  # Create a tensor e with values [1, 3, 5, 7, 9]

e = tf.cast(e, tf.float32)  # Cast tensor e to dtype float32

# Methametical operations

x = tf.constant([1,2,3])  # Create a tensor x with values [1, 2, 3]
y = tf.constant([9,8,7])  # Create a tensor y with values [9, 8, 7]

z = tf.add(x, y)  # Add x and y element-wise
z = x + y  # Same as tf.add(x, y)

z = tf.subtract(x, y)  # Subtract y from x element-wise
z = x - y  # Same as tf.subtract(x, y)

z = tf.divide(x, y)  # Divide x by y element-wise
z = x / y  # Same as tf.divide(x, y)

z = tf.multiply(x, y)  # Multiply x and y element-wise
z = x * y  # Same as tf.multiply(x, y)

z = tf.tensordot(x, y, axes=1)  # Compute the dot product of x and y
z = tf.reduce_sum(x * y, axis=0)  # Same as tf.tensordot(x, y, axes=1)

z = x ** 5  # Raise x to the power of 5

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))

z = tf.matmul(x, y)  # Matrix multiplication of tensors x and y

z = x @ y  # Same as tf.matmul(x, y)

# Indexing

x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
# print(x[:])
# print(x[1:])
# print(x[1:3])
# print(x[::2])
# print(x[::-1])

indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)

# print(x_ind)

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])

# print(x[0, :])
# print(x[0:2, :])
# print(x[0:2, 0])

# Reshaping

x = tf.range(9)

x = tf.reshape(x, (3,3))  # Reshape tensor x to shape (3,3)
x = tf.transpose(x, perm=[1,0])  # Transpose tensor x

print(x)
