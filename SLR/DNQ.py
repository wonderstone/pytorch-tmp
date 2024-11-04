import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# 定义强化学习中的 Q 网络
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 环境：简单模拟
class SimpleEnv:
    def __init__(self):
        self.x = np.linspace(-10, 10, 100).reshape(-1, 1)
        self.y = 2 * self.x + 3  # 目标回归函数
    
    def step(self, action):
        x = random.choice(self.x)
        true_y = 2 * x + 3
        reward = -np.abs(true_y - action)
        return x, reward, true_y

env = SimpleEnv()

# 初始化模型、损失函数和优化器
q_net = QNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(q_net.parameters(), lr=0.01)

# 模拟强化学习
epochs = 1000
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # 随机选择一个状态（x值）
    x = random.choice(env.x)
    x_tensor = torch.FloatTensor([x])

    # 使用 Q 网络预测值
    predicted_q = q_net(x_tensor)

    # 环境给出真实 y 值，并计算奖励
    true_y = 2 * x + 3
    reward = -np.abs(true_y - predicted_q.item())

    # 使用真实值计算目标 Q 值
    target_q = torch.FloatTensor([true_y])

    # 更新 Q 网络
    loss = criterion(predicted_q, target_q)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 可视化结果
q_net.eval()
predicted = [q_net(torch.FloatTensor([x])).item() for x in env.x]

plt.plot(env.x, env.y, label='True')
plt.plot(env.x, predicted, label='Predicted')
plt.legend()
plt.show()