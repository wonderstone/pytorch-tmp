import numpy as np

def masked_softmax(X, valid_lens):
    """
    X: 输入张量，shape (batch_size, seq_len, seq_len)
    valid_lens: 每个序列的有效长度，shape (batch_size,)
    """
    if valid_lens is None:
        return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = np.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        # 在最后一轴上遮蔽元素
        X = X.reshape(-1, shape[-1])
        maxlen = X.shape[1]
        mask = np.arange(maxlen)[None, :] < valid_lens[:, None]
        X[~mask] = -np.inf
        return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)

# 示例使用
X = np.random.randn(2, 3, 4)  # 批量大小为2，序列长度为3，每个位置的特征维度为4
valid_lens = np.array([2, 3])  # 第一个序列的有效长度为2，第二个为3

result = masked_softmax(X, valid_lens)
print(result)