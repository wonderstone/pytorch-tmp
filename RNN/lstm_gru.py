import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data
def generate_time_series(n_samples):
    time = np.linspace(0, 10, n_samples)
    series = np.sin(time) + np.random.normal(0, 0.1, n_samples)
    return series

# Prepare data for RNN/LSTM/GRU
def prepare_data(series, seq_length):
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])
    return np.array(X), np.array(y)

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden.squeeze(0))
        return out

# GRU Model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.gru(x)
        out = self.fc(hidden.squeeze(0))
        return out

# Generate and prepare data
n_samples = 1000
series = generate_time_series(n_samples)
seq_length = 50
X, y = prepare_data(series, seq_length)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X).unsqueeze(2)
y_tensor = torch.FloatTensor(y)

# Split into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Initialize models, loss function, and optimizers
lstm_model = LSTM(input_size=1, hidden_size=50, output_size=1)
gru_model = GRU(input_size=1, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # Train LSTM
    lstm_model.train()
    lstm_optimizer.zero_grad()
    lstm_output = lstm_model(X_train)
    lstm_loss = criterion(lstm_output.squeeze(), y_train)
    lstm_loss.backward()
    lstm_optimizer.step()
    
    # Train GRU
    gru_model.train()
    gru_optimizer.zero_grad()
    gru_output = gru_model(X_train)
    gru_loss = criterion(gru_output.squeeze(), y_train)
    gru_loss.backward()
    gru_optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], LSTM Loss: {lstm_loss.item():.4f}, GRU Loss: {gru_loss.item():.4f}')

# Evaluation
lstm_model.eval()
gru_model.eval()
with torch.no_grad():
    lstm_test_output = lstm_model(X_test)
    lstm_test_loss = criterion(lstm_test_output.squeeze(), y_test)
    gru_test_output = gru_model(X_test)
    gru_test_loss = criterion(gru_test_output.squeeze(), y_test)
    print(f'LSTM Test Loss: {lstm_test_loss.item():.4f}')
    print(f'GRU Test Loss: {gru_test_loss.item():.4f}')

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(y_test.numpy(), label='Actual')
plt.plot(lstm_test_output.squeeze().numpy(), label='LSTM Predicted')
plt.plot(gru_test_output.squeeze().numpy(), label='GRU Predicted')
plt.legend()
plt.title('Time Series Prediction using LSTM and GRU')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.show()