import torch
import torch.nn as nn
import math

# 定义嵌入层
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=512):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = self.create_position_encoding(max_len, embed_size)
    
    def create_position_encoding(self, max_len, embed_size):
        position_encoding = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                position_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_size)))
                position_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/embed_size)))
        return position_encoding.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.position_encoding[:, :seq_len, :].to(x.device)
        return x

# 多头注意力层示例
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_len, embed_size = x.shape

        # Linear projections to get Q, K, V
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # Reshape for multi-head attention
        queries = queries.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, values)

        # Concatenate heads and put through final linear layer
        out = out.transpose(1, 2).contiguous().view(N, seq_len, embed_size)
        return self.fc_out(out)

# 输入示例
vocab_size = 10000
embed_size = 512
num_heads = 8
seq_len = 10
batch_size = 32

# 初始化嵌入层和多头注意力层
embedding_layer = TransformerEmbedding(vocab_size, embed_size)
multi_head_attention = MultiHeadSelfAttention(embed_size, num_heads)

# 创建示例输入
input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))

# 通过嵌入和位置编码
embedded_input = embedding_layer(input_seq)
# 经过多头注意力
attention_output = multi_head_attention(embedded_input)
print("Attention Output Shape:", attention_output.shape)