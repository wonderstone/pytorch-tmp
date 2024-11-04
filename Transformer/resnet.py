import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        return F.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CustomResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.res_block1 = ResidualBlock(128, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.fc2(x)
        return x

# Example usage
input_size = 784  # e.g., for MNIST dataset (28x28 = 784)
num_classes = 10
model = CustomResNet(input_size, num_classes)

# Test the model
sample_input = torch.randn(1, input_size)
output = model(sample_input)
print(output.shape)  # Should print: torch.Size([1, 10])