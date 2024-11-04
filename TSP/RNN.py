import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
input_size = 10
hidden_size = 20
output_size = 5
seq_length = 15
batch_size = 3

# Create random input data
input_data = torch.randn(batch_size, seq_length, input_size)

# Initialize the model
model = SimpleRNN(input_size, hidden_size, output_size)

# Forward pass
output = model(input_data)
print(output.shape)  # Should be (batch_size, output_size)