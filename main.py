import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from charminal import *
from datetime import timedelta
# matplotlib.use('Agg')

def get_mnist(path):
  with np.load(path) as file:
    images, labels = file['x_train'], file['y_train']

  # images = images.astype('float32') / 255 # Normalize the image into [0, 1]
  print(f'{EMOJI_DEBUG} images shape before: {images.shape}')
  print(f'{EMOJI_DEBUG} labels shape before: {labels.shape}')
  images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2])) # Flatten the images from (28 * 28) to (784, 1)
  print(f'{EMOJI_DEBUG} images shape after: {images.shape}')

  labels = np.eye(10)[labels] # Make a one hot encoding by using an identity matrix with a size of (10 * 10)
  print(f'{EMOJI_DEBUG} labels shape after: {labels.shape}')

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
  print(f'{EMOJI_BEGIN} Retrieving dataset')
  start_time = time.time()
  images, labels = get_mnist('data/mnist.npz')
  print(f'{EMOJI_BEGIN} Successfully retrieved the dataset')
  print(f'{EMOJI_FINISH} Time elapsed: {timedelta(seconds=time.time() - start_time)}')

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

  print(f'{EMOJI_BEGIN} Start The Training Process')
  start_time = time.time()
  for epoch in range(epochs):
    for image, label in zip(images, labels):
      image.shape += (1,) # Convert from vector to a matrix --> (m,) to (m, 1)
      label.shape += (1,)

      # Forward propagation: Input --> Hidden
      hidden = w_i_h @ image + b_i_h
      hidden = sigmoid(hidden)

      # Forward propagation: Hidden --> Output
      output = w_h_o @ hidden + b_h_o
      output = sigmoid(output)

      # Calculate the cost
      cost = 1 / len(output) + np.sum((output - label) ** 2, axis=0)
      n_correct += int(np.argmax(output) == np.argmax(label)) # Check if output == label, if yes then n_correct += 1

      # Backpropagation: Output --> Hidden (cost function derivative)
      delta_output = output - label
      w_h_o += -learning_rate * delta_output @ np.transpose(hidden)
      b_h_o += -learning_rate * delta_output

      # Backpropagation: Hidden --> Input (activation function derivative)
      delta_hidden = np.transpose(w_h_o) @ delta_output * (hidden * (1 - hidden))
      w_i_h += -learning_rate * delta_hidden @ np.transpose(image)
      b_i_h += -learning_rate * delta_hidden

    # Show accuracy for this epoch
    print(f'+-------------------- EPOCH: {epoch} --------------------+')
    print(f'  >> Accuracy: {(n_correct / images.shape[0] * 100):.2f}%')
    print(f'  >> Time elapsed: {timedelta(seconds=time.time() - start_time)}')
    
    n_correct = 0
  
  print(f'{EMOJI_FINISH} Training complete!')
  print(f'{EMOJI_TIME} Time elapsed: {timedelta(seconds=time.time() - start_time)}')

  while True:
    index = int(input(f'\nEnter a number from 0 - 59999: '))
    image = images[index]
    label = labels[index]
    plt.imshow(image.reshape(28, 28), cmap='gray')

    image.shape += (1,)

    # Feed into the nets: Input --> Hidden
    hidden = w_i_h @ image.reshape(784, 1) + b_i_h
    hidden = sigmoid(hidden)
    # Feed into the nets: Hidden --> Output
    output = w_h_o @ hidden + b_h_o
    output = sigmoid(output)

    print(f'Output: {output}')
    print(f'Output (argmax): {np.argmax(output)}')
    print(f'Label: {label}')

    plt.title(f'Output: {output} | Label: {label}')
    plt.show()

if __name__ == '__main__':
  main()