import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data with 2 features
def generate_time_series(n_samples):
    time = np.linspace(0, 10, n_samples)
    series1 = np.sin(time) + np.random.normal(0, 0.1, n_samples)
    series2 = np.cos(time) + np.random.normal(0, 0.1, n_samples)
    return np.column_stack((series1, series2))

# Prepare data for RNN
def prepare_data(series, seq_length):
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])
    return np.array(X), np.array(y)

# RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out

# Generate and prepare data
n_samples = 1000
series = generate_time_series(n_samples)
seq_length = 50
X, y = prepare_data(series, seq_length)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)  # No need for unsqueeze here
y_tensor = torch.FloatTensor(y)

# Split into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Initialize model, loss function, and optimizer
model = RNN(input_size=2, hidden_size=50, output_size=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(y_test[:, 0].numpy(), label='Actual (Feature 1)')
plt.plot(test_output[:, 0].numpy(), label='Predicted (Feature 1)')
plt.plot(y_test[:, 1].numpy(), label='Actual (Feature 2)')
plt.plot(test_output[:, 1].numpy(), label='Predicted (Feature 2)')
plt.legend()
plt.title('Time Series Prediction using RNN (2 features)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.show()