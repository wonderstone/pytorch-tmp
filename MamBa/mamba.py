import torch
import torch.nn as nn

class MambaSSMCore(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(MambaSSMCore, self).__init__()
        
        # Transition model: Latent state evolution using LSTM
        self.transition_model = nn.LSTM(input_size=input_dim, hidden_size=latent_dim, batch_first=True)
        
        # Observation model: Maps latent state to output
        self.observation_model = nn.Linear(latent_dim, output_dim)
        
        # Initial hidden state
        self.latent_state_init = nn.Parameter(torch.randn(1, 1, latent_dim))
        
    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        
        # Expand the initial hidden state to batch size
        h_0 = self.latent_state_init.expand(1, batch_size, -1)  # Initial hidden state for LSTM
        c_0 = torch.zeros_like(h_0)  # Initial cell state for LSTM
        
        # State Equation: Evolve latent state over time
        latent_states, _ = self.transition_model(inputs, (h_0, c_0))
        
        # Output Equation: Map latent states to observable outputs
        outputs = self.observation_model(latent_states)
        
        return outputs, latent_states

# Define model parameters
input_dim = 10    # Input dimension (e.g., features in time series)
latent_dim = 20   # Latent state dimension
output_dim = 10   # Output dimension (e.g., predicted features)

# Initialize the Mamba core model
mamba_model = MambaSSMCore(input_dim, latent_dim, output_dim)

# Sample input data (batch_size, seq_len, input_dim)
sample_inputs = torch.randn(5, 15, input_dim)  # Example: batch of 5 sequences, each of length 15

# Run the model
outputs, latent_states = mamba_model(sample_inputs)
print("Outputs shape:", outputs.shape)          # (batch_size, seq_len, output_dim)
print("Latent states shape:", latent_states.shape)  # (batch_size, seq_len, latent_dim)