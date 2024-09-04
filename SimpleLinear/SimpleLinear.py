import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        # Initialize weight and bias as nn.Parameters
        self.weight = nn.Parameter(torch.tensor(0.0))  # Initial weight
        self.bias = nn.Parameter(torch.tensor(0.0))    # Initial bias

    def forward(self, x):
        return self.weight * x + self.bias

# Generate data
x_data = torch.linspace(-10, 10, 100)  # 100 points from -10 to 10
y_data = 0.7 * x_data + 0.3  # y = 0.7x + 0.3

# Initialize the model, loss function, and optimizer
model = SimpleLinearModel()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Training loop
num_epochs = 1000  # Number of epochs for training

for epoch in range(num_epochs):
    # ~ In PyTorch, this mode is crucial when using certain layers like 
    # ~ dropout or batch normalization that behave differently during training and evaluation. 
    # ~ In this simple model, it’s more for convention, 
    # ~ as there are no layers that behave differently in training mode.
    model.train()  # Set the model to training mode
    
    # Forward pass
    y_pred = model(x_data)
    
    # Compute loss
    loss = criterion(y_pred, y_data)
    
    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the progress
    if (epoch+1) % 100 == 0:
        # Set the model to evaluation mode before printing (not strictly necessary for this simple model)
        model.eval()
        with torch.no_grad():  # Disable gradient computation for evaluation
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, w: {model.weight.item():.4f}, b: {model.bias.item():.4f}')

# Final evaluation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    print(f'\nFinal weight: {model.weight.item():.4f}')
    print(f'Final bias: {model.bias.item():.4f}')