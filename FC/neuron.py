# Neuron: Single neuron implementation with forward and backward pass

# Import the necessary libraries
import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define a single neuron
class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias  # Linear combination
        self.a = sigmoid(self.z)  # Activation (Sigmoid)
        return self.a

    def backward(self, error, learning_rate):
        # Gradient of the activation function
        d_a = error * sigmoid_derivative(self.a)

        # Gradients for weights and bias
        d_weights = np.dot(self.inputs.T, d_a)
        d_bias = d_a

        # Update weights and bias
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * np.sum(d_bias)

if __name__ == "__main__":
    # Training loop for a single neuron
    inputs = np.array([1.0, 2.0, 3.0])  # Input example
    target = np.array(0.0)              # Target output
    neuron = Neuron(num_inputs=3)        # Create the neuron
    learning_rate = 0.1

    # Training
    for epoch in range(10000):
        # Forward pass
        output = neuron.forward(inputs)

        # Calculate the error (target - output)
        error = abs(target - output)

        # Backward pass (gradient calculation and weight update)
        neuron.backward(error, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Output: {output}, Error: {error}")

    print("Final output:", neuron.forward(inputs))

