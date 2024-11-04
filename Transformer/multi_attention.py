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

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.fc_out = torch.nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, query, key, value):
        N = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        # Split embedding into self.num_heads different pieces
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)
        key = key.reshape(N, key_len, self.num_heads, self.head_dim)
        value = value.reshape(N, value_len, self.num_heads, self.head_dim)

        # Self-Attention on each head
        attention = basic_attention(query, key, value)
        
        # Concatenate heads
        out = attention.reshape(N, query_len, self.num_heads * self.head_dim)
        return self.fc_out(out)

# 使用Multi-Head Attention
embed_size = 64
num_heads = 8
multi_head_attention = MultiHeadAttention(embed_size, num_heads)

query = torch.randn(1, 3, embed_size)
key = torch.randn(1, 3, embed_size)
value = torch.randn(1, 3, embed_size)

output = multi_head_attention(query, key, value)
print("Multi-Head Attention Output Shape:", output.shape)