import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neuron using nn.Module
class Neuron(nn.Module):
    def __init__(self, num_inputs):
        super(Neuron, self).__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))

if __name__ == "__main__":
    # Create the neuron
    neuron = Neuron(num_inputs=3)

    # Define loss function (Mean Squared Error) and optimizer (SGD)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(neuron.parameters(), lr=0.1)

    # Training data
    inputs = torch.tensor([1.0, 2.0, 3.0])  # Input example
    target = torch.tensor([0.0])            # Target output

    # Training loop
    for epoch in range(1000):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = neuron(inputs)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass (automatic gradient computation)
        loss.backward()

        # Update weights using optimizer
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Output: {output.item()}, Loss: {loss.item()}")

    print("Final output:", neuron(inputs).item())