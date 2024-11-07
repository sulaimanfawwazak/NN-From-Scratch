import numpy as np
import matplotlib.pyplot as plt

def get_mnist(path):
  with np.load(path) as file:
    images, labels = file['x_train'], file['y_train']

  # images = images.astype('float32') / 255 # Normalize the image into [0, 1]
  print(f'images shape before: {images.shape}')
  images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2])) # Flatten the images from (28 * 28) to (784, 1)
  print(f'images shape after: {images.shape}')

  labels = np.eye(10)[labels] # Make a one hot encoding by using an identity matrix with a size of (10 * 10)

  return images, labels

def sigmoid(value):
  return 1 / (1 + np.exp(-value))

"""
w: weights, b: bias, i: input, o: output, l: label

Example:
w_i_h: weights from input layer to hidden layer
"""

def main():
  # Retrieve the images and labels
  images, labels = get_mnist('data/mnist.npz')
  images = images / 255.0 # Normalize the image

  # Initialize the weights of the layers with random values between [-0.5, 0.5]
  w_i_h = np.random.uniform(-0.5, 0.5, (20, 784)) # Create the input layer to hidden layer; We specify the shape as (next, current)
  w_h_o = np.random.uniform(-0.5, 0.5, (10, 20)) # Create the hidden layer to output layer
  
  # Initialize the bias for each layers with a value of 0
  b_i_h = np.zeros((20, 1))
  b_h_o = np.zeros((10, 1))

  # Hyperparameters
  learning_rate = 0.01
  n_correct = 0
  epochs = 3

  for epoch in range(epochs):
    for image, label in zip(images, labels):
      image.shape += (1,) # Convert from vector to a matrix --> (m,) to (m, 1)
      label.shape += (1,)

      # Forward propagation: Input --> Hidden
      h = w_i_h @ image + b_i_h
      h = sigmoid(h)

      # Forward propagation: Hidden --> Output
      o = w_h_o @ h + b_h_o
      o = sigmoid(o)

      # Calculate the error


if __name__ == '__main__':
  main()