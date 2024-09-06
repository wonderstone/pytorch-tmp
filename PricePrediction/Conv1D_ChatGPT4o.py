import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成示例数据
def generate_data(seq_length, n_samples):
    np.random.seed(42)
    x = np.linspace(0, 100, n_samples)
    y = np.sin(x) + np.random.normal(0, 0.1, n_samples)  # 添加噪声的价格数据
    data = []
    for i in range(len(x) - seq_length):
        data.append(y[i:i + seq_length])
    return np.array(data), y[seq_length:]

seq_length = 5
n_samples = 200
X, y = generate_data(seq_length, n_samples)

# 2. 数据预处理并转换为PyTorch张量
X_train = torch.FloatTensor(X).unsqueeze(1)  # (batch_size, 1, seq_length)
y_train = torch.FloatTensor(y).unsqueeze(-1)  # (batch_size, 1)

# 3. 定义1D CNN模型
class CNN1DTimeSeries(nn.Module):
    def __init__(self):
        super(CNN1DTimeSeries, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)  # Conv1D层
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        
        # 计算全连接层输入大小
        self.flatten_size = self._get_conv_output_shape()

        self.fc1 = nn.Linear(self.flatten_size, 50)  # 全连接层1
        self.fc2 = nn.Linear(50, 1)  # 最终输出为单一价格预测

    def _get_conv_output_shape(self):
        # 用随机张量推导卷积层的输出形状
        with torch.no_grad():
            x = torch.zeros(1, 1, seq_length)  # (batch_size=1, in_channels=1, seq_length)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return x.numel()  # 计算展平后的元素数量

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 卷积1
        x = torch.relu(self.conv2(x))  # 卷积2
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))  # 全连接层1
        return self.fc2(x)  # 输出层

# 4. 初始化模型、损失函数和优化器
model = CNN1DTimeSeries()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
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

# 6. 绘制训练损失曲线
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 7. 测试模型并可视化结果
model.eval()
predicted = model(X_train).detach().numpy()

# 8. 可视化实际价格与预测价格
plt.plot(y, label='True Prices')
plt.plot(predicted, label='Predicted Prices')
plt.legend()
plt.show()
pass