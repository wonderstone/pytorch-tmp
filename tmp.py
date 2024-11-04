import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# give a tensor in shape of (2, 3, 4)

tmp = torch.randn(2, 3, 4)

print(tmp)

tmp1 = tmp[-1, :, :]
print(tmp1)