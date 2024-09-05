import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * x + 3  # 目标函数

# 转换为 PyTorch 张量
x_train = torch.FloatTensor(x)
y_train = torch.FloatTensor(y)

# 定义一个简单的全连接网络
class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 初始化模型、损失函数和优化器
model = SimpleFC()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练网络
epochs = 1000
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 可视化结果
model.eval()
predicted = model(x_train).detach().numpy()

plt.plot(x, y, label='True')
plt.plot(x, predicted, label='Predicted')
plt.legend()
plt.show()