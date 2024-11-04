import torch
import torch.nn.functional as F

def masked_softmax(X, mask):
    """
    Compute softmax with a mask, ensuring masked positions have a near-zero probability.
    
    Parameters:
    - X: Input tensor (batch_size, seq_len)
    - mask: Mask tensor with same shape as X, where 0 = masked, 1 = unmasked (or bool mask)

    Returns:
    - Tensor with softmax applied to the unmasked values only.
    """
    # Replace masked positions with a large negative value
    X = X.masked_fill(~mask, -1e9)  # ~mask is the inverse (mask with 0s for positions to ignore)
    
    # Apply softmax
    softmax_output = F.softmax(X, dim=-1)
    
    return softmax_output

# Example usage
batch_size, seq_len = 2, 5
X = torch.randn(batch_size, seq_len)

# Create a mask with 1s for active positions and 0s for masked (e.g., padding or future tokens)
mask = torch.tensor([[1, 1, 1, 0, 0],  # mask last two tokens
                     [1, 1, 0, 0, 0]]) # mask last three tokens
mask = mask.bool()  # Convert to boolean mask

masked_softmax_output = masked_softmax(X, mask)
print("Masked Softmax Output:\n", masked_softmax_output)