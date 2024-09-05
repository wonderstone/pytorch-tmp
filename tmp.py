import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Download stock price data
ticker = 'AAPL'  # Example: Apple stock
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
prices = data['Adj Close'].values  # Use adjusted closing prices

# 2. Prepare the data: create sliding windows of length 5
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

seq_length = 5
X, y = create_sequences(prices, seq_length)

# Convert to PyTorch tensors and reshape for Conv1D (batch_size, seq_length, num_features)
X_train = torch.FloatTensor(X).unsqueeze(-1)  # (batch_size, seq_length, 1)
y_train = torch.FloatTensor(y).unsqueeze(-1)  # (batch_size, 1)

# 3. Define the 1D CNN model for time series prediction
class CNN1DTimeSeries(nn.Module):
    def __init__(self):
        super(CNN1DTimeSeries, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.fc1 = nn.Linear(32 * (seq_length - 3), 50)  # Adjust for Conv1D output
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 4. Initialize model, loss function, and optimizer
model = CNN1DTimeSeries()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train the model
epochs = 100
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Plot the training loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 6. Make predictions and visualize the results
model.eval()
predicted = model(X_train).detach().numpy()

# Visualize true prices vs predicted prices
plt.plot(y, label='True Prices')
plt.plot(predicted, label='Predicted Prices')
plt.legend()
plt.show()