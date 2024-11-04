import torch
import torch.nn as nn

class RecurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RecurrentLayer, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size)
        
        outputs = []
        for i in range(x.size(1)):
            hidden = self.rnn_cell(x[:, i, :], hidden)
            outputs.append(hidden)
        
        return torch.stack(outputs, dim=1), hidden

# Example usage
input_size = 10
hidden_size = 20
seq_length = 5
batch_size = 3

# Create random input data
input_data = torch.randn(batch_size, seq_length, input_size)

# Initialize the recurrent layer
recurrent_layer = RecurrentLayer(input_size, hidden_size)

# Forward pass
output, final_hidden = recurrent_layer(input_data)
print(output.shape)  # Should be (batch_size, seq_length, hidden_size)
print(final_hidden.shape)  # Should be (batch_size, hidden_size)