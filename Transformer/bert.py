import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简化的BERT模型
class SimpleBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes):
        super(SimpleBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 简单的池化操作
        return self.fc(x)

# 定义超参数
vocab_size = 10000
embed_dim = 256
num_heads = 4
num_layers = 2
num_classes = 2
max_seq_length = 128

# 初始化模型
model = SimpleBERT(vocab_size, embed_dim, num_heads, num_layers, num_classes)

# 准备一些模拟数据
batch_size = 16
x = torch.randint(0, vocab_size, (batch_size, max_seq_length))
y = torch.randint(0, num_classes, (batch_size,))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 简单的训练循环
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 使用模型进行预测
with torch.no_grad():
    test_input = torch.randint(0, vocab_size, (1, max_seq_length))
    prediction = model(test_input)
    predicted_class = torch.argmax(prediction, dim=1)
    print(f"Predicted class: {predicted_class.item()}")