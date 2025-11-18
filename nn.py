import argparse
import gzip
import numpy as np
import pandas as pd
import random

#The sigmoid activation function.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#Derivative of the sigmoid function.
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

"""Loads the Fashion-MNIST dataset from gzipped CSV files and formats it
    for the neural network."""
def load_data_wrapper(train_file, test_file):
    print("Loading and preparing data...")
    df_train = pd.read_csv(train_file, compression='gzip')
    df_test = pd.read_csv(test_file, compression='gzip')

    train_labels = df_train.iloc[:, 0].values
    train_images = df_train.iloc[:, 1:].values
    
    train_images = train_images / 255.0
    
    training_inputs = [np.reshape(x, (784, 1)) for x in train_images]
    training_results = [vectorized_result(y) for y in train_labels]
    training_data = list(zip(training_inputs, training_results))

    test_labels = df_test.iloc[:, 0].values
    test_images = df_test.iloc[:, 1:].values
    test_images = test_images / 255.0
    
    testing_inputs = [np.reshape(x, (784, 1)) for x in test_images]
    test_data = list(zip(testing_inputs, test_labels))
    
    print("Data loading complete.")
    return (training_data, test_data)

"""Returns a 10-dimensional one-hot encoded vector with a 1.0 at the j-th position.
    This is used to format the training labels."""
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

""" A three-layer neural network. The core logic for feedforward prediction and
    backpropagation training is encapsulated in this class."""
class NeuralNetwork:
    """ Initializes the network.
        'sizes' is a list containing the number of neurons in each layer.
        Example: [784, 30, 10] for an input layer of 784, hidden of 30, output of 10. """
    def __init__(self, sizes):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    """ Passes an input 'a' through the network and returns the final output."""
    def feed_forward(self, a):
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

        """Trains the neural network using mini-batch stochastic gradient descent (SGD).
        
        Args:
            training_data: A list of tuples (x, y) representing inputs and desired outputs.
            epochs: The number of times to loop over the entire training set.
            mini_batch_size: The number of training examples to use in each batch.
            eta: The learning rate (η).
            test_data: Optional. If provided, the network's accuracy will be evaluated
                       on this data after each epoch. """
    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
       
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                n_test = len(test_data)
                accuracy = self.evaluate(test_data)
                print(f"Epoch {j}: {accuracy} / {n_test} ({accuracy / n_test:.2%})")
            else:
                print(f"Epoch {j} complete")

    """Updates the network's weights and biases by applying gradient descent
        using backpropagation to a single mini-batch."""
    def update_mini_batch(self, mini_batch, eta):
        # Initialize accumulators for the gradients with zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # For each training example in the batch...
        for x, y in mini_batch:
            # ...calculate the gradients using backpropagation
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # ...and add them to the accumulators
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # Update the actual weights and biases using the averaged gradients
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

        """"
        Performs backpropagation for a single training example (x, y).
        Returns a tuple (nabla_b, nabla_w) representing the gradient of the
        cost function for this single example.
        """
    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # We need to store all the activations and z-vectors (pre-activation)
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # Calculate the error for the output layer
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Backpropagate the error to the previous layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            
        return (nabla_b, nabla_w)

        """
        Evaluates the network's performance on the test data.
        Returns the number of correctly classified images.
        """
    def evaluate(self, test_data):
        
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    """
    Parses command-line arguments and runs the neural network training.
    """
def main():
  
    parser = argparse.ArgumentParser(description="Train a 3-layer neural network on Fashion-MNIST.")
    parser.add_argument('train_file', type=str, help="Path to the training data file (e.g., fashion-mnist_train.csv.gz)")
    parser.add_argument('test_file', type=str, help="Path to the test data file (e.g., fashion-mnist_test.csv.gz)")
    parser.add_argument('--hidden', type=int, default=30, help="Number of neurons in the hidden layer (default: 30)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument('--batch_size', type=int, default=20, help="Mini-batch size for training (default: 20)")
    parser.add_argument('--learning_rate', type=float, default=3.0, help="Learning rate (eta) (default: 3.0)")
    
    args = parser.parse_args()

    training_data, test_data = load_data_wrapper(args.train_file, args.test_file)

    # Define the network architecture: Input -> Hidden -> Output
    # Input layer has 784 neurons (28x28 pixels).
    # Output layer has 10 neurons (for 10 classes).
    net = NeuralNetwork([784, args.hidden, 10])

    print(f"\n--- Starting Training ---")
    print(f"Hidden Neurons: {args.hidden}, Epochs: {args.epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}")
    net.train(training_data, args.epochs, args.batch_size, args.learning_rate, test_data=test_data)
    print("--- Training Complete ---")

if __name__ == "__main__":
    main()