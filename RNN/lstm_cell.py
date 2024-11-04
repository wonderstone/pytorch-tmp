import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create a list of LSTMCell layers
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size, hidden_size)])
        self.lstm_cells.extend([nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state and cell state
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        
        # Process each time step
        outputs = []
        for t in range(seq_length):
            batch_x = x[:, t, :]  # Extract the t-th time step
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer], c[layer] = self.lstm_cells[layer](batch_x, (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.lstm_cells[layer](h[layer-1], (h[layer], c[layer]))
            outputs.append(h[-1].unsqueeze(1))
        
        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)
        
        # Use the final hidden state for prediction
        output = self.fc(h[-1])
        return output

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
seq_length = 15
batch_size = 32

# Create the model
model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)

# Generate sample input data
input_data = torch.randn(batch_size, seq_length, input_size)
lengths = torch.randint(1, seq_length + 1, (batch_size,))

# Forward pass
output = model(input_data, lengths)
print("Output shape:", output.shape)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop (example)
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data, lengths)
    
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
    test_lengths = torch.tensor([seq_length])
    prediction = model(test_input, test_lengths)
    print("Prediction shape:", prediction.shape)