import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with shapes based on conventional RNNs
        self.Wxh = np.random.randn(self.hidden_size, self.input_size) * 0.01  # Input to hidden weights
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01  # Hidden to hidden weights
        self.Why = np.random.randn(self.output_size, self.hidden_size) * 0.01  # Hidden to output weights
        self.bh = np.zeros((self.hidden_size, 1))  # Hidden layer bias
        self.by = np.zeros((self.output_size, 1))  # Output layer bias

    def forward(self, inputs):
        """Forward pass through time for a sequence of inputs."""
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        self.last_inputs = inputs
        self.hidden_states = [h]  # Save hidden states
        
        # Iterate through the time steps
        for i in range(len(inputs)):
            x = inputs[i].reshape(-1, 1)  # Shape (input_size, 1)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)  # Shape (hidden_size, 1)
            self.hidden_states.append(h)
        
        # Final output (using last hidden state)
        y = np.dot(self.Why, h) + self.by  # Shape (output_size, 1)
        return y, h

    def backward(self, d_y):
        """Backward pass through time (BPTT)"""
        d_Why = np.dot(d_y, self.hidden_states[-1].T)
        d_by = d_y

        d_Whh = np.zeros_like(self.Whh)
        d_Wxh = np.zeros_like(self.Wxh)
        d_bh = np.zeros_like(self.bh)
        
        # Gradient of loss with respect to last hidden state
        d_h = np.dot(self.Why.T, d_y)
        
        for t in reversed(range(len(self.last_inputs))):
            temp_h = self.hidden_states[t+1]
            d_z = (1 - temp_h ** 2) * d_h  # Derivative of tanh
            
            # Calculate gradients for the weights and biases
            d_Whh += np.dot(d_z, self.hidden_states[t].T)
            d_Wxh += np.dot(d_z, self.last_inputs[t].reshape(-1, 1).T)
            d_bh += d_z
            
            # Update d_h for the previous time step
            d_h = np.dot(self.Whh.T, d_z)
        
        # Update weights and biases
        for param, d_param in zip([self.Whh, self.Wxh, self.Why, self.bh, self.by],
                                  [d_Whh, d_Wxh, d_Why, d_bh, d_by]):
            param -= self.learning_rate * d_param

# Usage
input_size = 1
hidden_size = 5
output_size = 1
learning_rate = 0.01

# Initialize RNN model
rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate)

# Generate example sequential data
inputs = [np.array([i]) for i in range(5)]
target = np.array([[0.5]])

# Training example
for epoch in range(100):
    y_pred, h = rnn.forward(inputs)
    loss = np.square(y_pred - target).sum() / 2
    d_y = y_pred - target  # Gradient of loss with respect to output
    
    # Backpropagation through time
    rnn.backward(d_y)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")