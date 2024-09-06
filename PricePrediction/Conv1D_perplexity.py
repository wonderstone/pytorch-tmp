import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Generate synthetic time series data
np.random.seed(0)
time_steps = 1000
data = np.sin(np.linspace(0, 100, time_steps)) + np.random.normal(scale=0.5, size=time_steps)

# Create a DataFrame
df = pd.DataFrame(data, columns=['Price'])

# Normalize the data
scaler = MinMaxScaler()
df['Price'] = scaler.fit_transform(df[['Price']])

# Define window size and horizon
WINDOW_SIZE = 7
HORIZON = 1

# Function to create windowed data
def create_windows(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1])
    return np.array(X), np.array(y)

# Create windowed data
X, y = create_windows(df['Price'].values, WINDOW_SIZE, HORIZON)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * WINDOW_SIZE, 1)  # Adjust based on output size after convolutions

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x
    
# Instantiate the model, define loss function and optimizer
model = Conv1DModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Inverse transform to get actual prices
predictions = scaler.inverse_transform(predictions.numpy())
y_test_actual = scaler.inverse_transform(y_test.numpy())

# Display predictions vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.title('Conv1D Model Predictions vs Actual Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()