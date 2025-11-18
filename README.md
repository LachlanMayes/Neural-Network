---

# Neural Network for Fashion-MNIST from Scratch

This project provides a from-scratch implementation of a feedforward neural network in Python. It is designed to classify images from the Fashion-MNIST dataset using stochastic gradient descent (SGD) and the backpropagation algorithm. The network's architecture and training hyperparameters can be easily configured via command-line arguments.

## Key Features

*   **Pure Python & NumPy**: The core neural network logic is implemented using only standard Python libraries and NumPy, without relying on high-level frameworks like TensorFlow or PyTorch.
*   **Backpropagation Algorithm**: The network is trained using the backpropagation algorithm to efficiently compute the gradient of the cost function.
*   **Stochastic Gradient Descent (SGD)**: The model's weights and biases are optimized using mini-batch SGD.
*   **Customizable Architecture**: The number of neurons in the hidden layer can be specified at runtime.
*   **Configurable Hyperparameters**: Key training parameters such as epochs, mini-batch size, and learning rate can be easily adjusted through command-line arguments.
*   **Data Handling**: Includes functions to load, parse, and preprocess the gzipped Fashion-MNIST CSV files into a format suitable for the network.

## How It Works

The script follows the fundamental principles of training a neural network:

1.  **Initialization**: The network is initialized with a given architecture (e.g., `[784, 30, 10]` for an input layer, a hidden layer of 30 neurons, and an output layer). Weights and biases are assigned random initial values.
2.  **Data Loading**: The Fashion-MNIST dataset is loaded from gzipped CSV files. Pixel values are normalized to a range of `0.0` to `1.0`, and labels are one-hot encoded.
3.  **Training Loop**: The network is trained for a specified number of epochs. In each epoch:
    *   The training data is shuffled and split into mini-batches.
    *   For each mini-batch, the network applies the backpropagation algorithm to calculate the gradients for the weights and biases.
    *   The weights and biases are updated using the calculated gradients and the learning rate (gradient descent step).
4.  **Feedforward**: To make a prediction, an input image is passed through the network layer by layer. At each neuron, a weighted sum of the inputs from the previous layer is calculated, the bias is added, and the result is passed through a sigmoid activation function.
5.  **Evaluation**: After each epoch, the network's performance is evaluated on the test dataset. The output neuron with the highest activation is chosen as the predicted class, and the overall accuracy is calculated and displayed.

## Getting Started

### Prerequisites

*   Python 3.x
*   The following Python libraries:
    *   NumPy
    *   Pandas

### Installation

1.  Clone the repository:
    ```bash
    git clone (https://github.com/LachlanMayes/Neural-Network)
    ```

2.  Install the required libraries using pip:
    ```bash
    pip install numpy pandas
    ```

3.  **Download the Dataset**: This script is designed to work with the Fashion-MNIST dataset in a gzipped CSV format. You can download it from Kaggle:
    *   **[Fashion-MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)**
    *   Download the `fashion-mnist_train.csv.gz` and `fashion-mnist_test.csv.gz` files and place them in the same directory as the `nn.py` script.

### Running the Script

You can run the script from your terminal. You must provide the paths to the training and test data files. You can also specify optional arguments to customize the training process.

**Basic Usage:**

```bash
python nn.py fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz
```

**Customized Training:**

To change the network architecture or training hyperparameters, use the optional flags:

```bash
python nn.py fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz --hidden 100 --epochs 30 --batch_size 10 --learning_rate 3.0
```

#### Command-Line Arguments

*   `train_file`: (Required) Path to the training data file (e.g., `fashion-mnist_train.csv.gz`).
*   `test_file`: (Required) Path to the test data file (e.g., `fashion-mnist_test.csv.gz`).
*   `--hidden`: Number of neurons in the hidden layer. Default is `30`.
*   `--epochs`: Number of training epochs. Default is `10`.
*   `--batch_size`: The size of mini-batches for SGD. Default is `20`.
*   `--learning_rate`: The learning rate (eta) for training. Default is `3.0`.

## Expected Output

When you run the script, it will first load the data and then begin the training process. After each epoch, it will print the current accuracy on the test set:

```
Loading and preparing data...
Data loading complete.

--- Starting Training ---
Hidden Neurons: 30, Epochs: 10, Batch Size: 20, Learning Rate: 3.0
Epoch 0: 8254 / 10000 (82.54%)
Epoch 1: 8382 / 10000 (83.82%)
Epoch 2: 8448 / 10000 (84.48%)
...
Epoch 9: 8521 / 10000 (85.21%)
--- Training Complete ---
```
