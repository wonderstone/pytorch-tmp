# FC.py is a simple fully connected neural network for classifying MNIST digits.
# FC layer expects a 1D input, so we flatten the 28x28 input image to a 784 vector.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.draw import DynamicPlot

# Define the network architecture
class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input from 28x28 to 784
        # ~ X shape at this point: [batch_size, 784]
        x = self.relu(self.fc1(x))
        # ~ X shape at this point: [batch_size, 128]
        x = self.relu(self.fc2(x))
        # ~ X shape at this point: [batch_size, 64]
        x = self.fc3(x)
        # ~ X shape at this point: [batch_size, 10]
        return x

if __name__ == "__main__":
    plotter = DynamicPlot()
    plotter.add_line('train-loss', 'red')
    # Set up data loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SimpleFC()
    # @ CrossEntropyLoss: This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    # 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                # ~ plot loss
                x_val = len(train_loader)*epoch+batch_idx
                y_val = loss.item()
                plotter.add_data('train-loss', x_val, y_val)
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    plotter.show()
    pass


