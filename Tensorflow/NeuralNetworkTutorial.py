import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import sys 
import io


"""
This script demonstrates how to build and train a neural network model using TensorFlow and Keras.
It loads the MNIST dataset, preprocesses the data, builds a neural network model using the Functional API,
compiles the model with a loss function, optimizer, and metrics, trains the model on the training data,
evaluates the model on the test data, and predicts the output for a sample test data point.
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

"""
model = keras.Sequential(
    [
        keras.Input(shape=(784,)),  # Input layer for flattened 28x28 images
        layers.Dense(512, activation='relu'),  # First hidden layer
        layers.Dense(256, activation='relu'),  # Second hidden layer
        layers.Dense(10),  # Output layer for 10 classes
    ]
)
"""

# Define the model architecture
inputs = keras.Input(shape=(784,))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Print the model summary
print(model.summary())

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# Evaluate the model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# Make predictions
model.predict(x_test[:1])
