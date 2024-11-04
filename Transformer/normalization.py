import torch
import torch.nn as nn

# Sample input data
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Batch Normalization across batch dimension
batch_norm = nn.BatchNorm1d(num_features=3, affine=False)
batch_norm_output = batch_norm(X)

# Layer Normalization across feature dimension
layer_norm = nn.LayerNorm(normalized_shape=3, elementwise_affine=False)
layer_norm_output = layer_norm(X)

print("Input:\n", X)
print("\nBatch Normalized Output:\n", batch_norm_output)
print("\nLayer Normalized Output:\n", layer_norm_output)



import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, feature_dim, epsilon=1e-5):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(feature_dim))  # Shift parameter
        self.epsilon = epsilon

    def forward(self, x):
        # Calculate mean and variance across the last dimension (feature dimension)
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        return output

# Example usage
# Suppose we have a batch of data with shape (batch_size, seq_len, feature_dim)
batch_size, seq_len, feature_dim = 3, 4, 5
x = torch.randn(batch_size, seq_len, feature_dim)

layer_norm = LayerNormalization(feature_dim)
output = layer_norm(x)

print("Input:", x)
print("Output after LayerNorm:", output)

layer_norm1 = LayerNormalization(3)
output1 = layer_norm1(X)
print("Output after LayerNorm:", output1)
