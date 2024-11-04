import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # ~ Case 1
        # # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])

        # ~ Case 2
        # Forward propagate LSTM
        _, hidden = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(hidden[0][-1, :, :])


        return out

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
seq_length = 15
batch_size = 32

# Create the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Generate sample input data
input_data = torch.randn(batch_size, seq_length, input_size)

# Forward pass
output = model(input_data)
print("Output shape:", output.shape)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop (example)
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)
    
    # Compute loss (assuming we have target data)
    target = torch.randn(batch_size, output_size)  # Replace with actual target data
    loss = criterion(outputs, target)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the model for predictions
with torch.no_grad():
    test_input = torch.randn(1, seq_length, input_size)
    prediction = model(test_input)
    print("Prediction shape:", prediction.shape)