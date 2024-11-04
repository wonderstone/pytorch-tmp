import numpy as np

# Step 1: Define input sequence
sequence = ["I", "love", "machine", "learning"]
seq_len = len(sequence)
d_model = 4  # Embedding dimension

# Step 2: Create random embeddings for each word
embeddings = np.random.rand(seq_len, d_model)

# Step 3: Define weight matrices for Q, K, V
W_q = np.random.rand(d_model, d_model)
W_k = np.random.rand(d_model, d_model)
W_v = np.random.rand(d_model, d_model)

# Step 4: Calculate Q, K, V
Q = np.dot(embeddings, W_q)
K = np.dot(embeddings, W_k)
V = np.dot(embeddings, W_v)

# Step 5: Calculate attention scores
scores = np.dot(Q, K.T) / np.sqrt(d_model)

# Step 6: Apply softmax to get attention weights
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

# Step 7: Calculate the output
output = np.dot(attention_weights, V)

print("Input embeddings:")
print(embeddings)
print("\nAttention weights:")
print(attention_weights)
print("\nOutput:")
print(output)



import torch
import torch.nn.functional as F

def basic_attention(query, key, value):
    # 计算分数（查询和键的内积）
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(key.shape[-1], dtype=torch.float32))
    # 使用Softmax获得权重
    weights = F.softmax(scores, dim=-1)
    # 加权求和值
    attention_output = torch.matmul(weights, value)
    return attention_output

# 示例输入
query = torch.randn(1, 3, 64)  # (batch_size, seq_len, d_k)
key = torch.randn(1, 3, 64)
value = torch.randn(1, 3, 64)

output = basic_attention(query, key, value)
print("Basic Attention Output Shape:", output.shape)