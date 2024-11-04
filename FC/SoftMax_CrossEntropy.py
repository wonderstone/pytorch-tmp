# The relation between SoftMax and CrossEntropy
# SoftMax: y = e^x / sum(e^x)
# CrossEntropy: -sum(y_true * log(y_pred))
# The following code snippet demonstrates:
# CrossEntropyLoss = -log(softmax(logits)[target])
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create sample data
logits = torch.tensor([[2.0, 1.0, 0.1]])  # Raw output from the network
target = torch.tensor([0])  # True class (index)

print("Step 1: Raw logits")
print(logits)

# Step 2: Apply softmax
probabilities = F.softmax(logits, dim=1)
print("\nStep 2: After softmax")
print(probabilities)

# Step 3: Take log of probabilities
log_probabilities = torch.log(probabilities)
print("\nStep 3: Log probabilities")
print(log_probabilities)

# Step 4: Select the log probability of the true class
selected_log_prob = log_probabilities[0, target]
print("\nStep 4: Selected log probability")
print(selected_log_prob)

# Step 5: Negative log likelihood (NLL)
nll_loss = -selected_log_prob
print("\nStep 5: Negative log likelihood")
print(nll_loss)

# Verify with PyTorch's built-in functions
ce_loss = nn.CrossEntropyLoss()(logits, target)
print("\nPyTorch's CrossEntropyLoss")
print(ce_loss)

# Compare with manual calculation
manual_ce_loss = -torch.log(probabilities[0, target])
print("\nManually calculated Cross Entropy Loss")
print(manual_ce_loss)

assert torch.isclose(nll_loss, ce_loss), "NLL loss should be equal to CrossEntropyLoss"
assert torch.isclose(manual_ce_loss, ce_loss), "Manual calculation should match PyTorch's CrossEntropyLoss"
print("\nAll assertions passed. The manual calculation matches PyTorch's implementation.")