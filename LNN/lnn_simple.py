import torch
import torch.nn as nn
import torch.optim as optim

# Define the Liquid Neural Network Layer
class LiquidNeuronLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LiquidNeuronLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.dynamic_weight = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, hidden_state):
        # Calculate input and hidden influences
        input_influence = torch.tanh(self.input_to_hidden(x))
        hidden_influence = torch.tanh(self.hidden_to_hidden(hidden_state))

        # Modify hidden state dynamically with 'liquid' parameter
        dynamic_influence = hidden_influence * torch.sigmoid(self.dynamic_weight)
        new_hidden_state = input_influence + dynamic_influence
        return new_hidden_state, new_hidden_state

# Define the full Liquid Neural Network model
class LiquidNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LiquidNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.liquid_layer = LiquidNeuronLayer(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # Process each time step
        for t in range(seq_len):
            hidden_state, _ = self.liquid_layer(x[:, t, :], hidden_state)
        
        # Pass the last hidden state to output layer
        output = self.output_layer(hidden_state)
        return output

# Initialize and test the model
input_dim = 10   # Input feature size
hidden_dim = 20  # Size of the liquid neuron layer
output_dim = 1   # Output size (e.g., regression or binary classification)

model = LiquidNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()  # Loss function (mean squared error)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy data (batch_size, seq_len, input_dim)
x = torch.randn(5, 15, input_dim)  # Example input
y = torch.randn(5, output_dim)     # Example target output

# Forward pass
output = model(x)
loss = criterion(output, y)

# Backward pass and optimization
loss.backward()
optimizer.step()

print("Output:", output)
print("Loss:", loss.item())