import torch
import torch.nn as nn

# Create an Embedding layer
vocab_size = 100  # Number of unique tokens
embedding_dim = 5  # Size of each embedding vector
embedding = nn.Embedding(vocab_size, embedding_dim)

# Input: a tensor of indices
input_indices = torch.LongTensor([1, 4, 3, 2])

# Get embeddings for these indices
embedded = embedding(input_indices)

print(embedded.shape)  # Output: torch.Size([4, 5])
print(embedded)
# Output:
# tensor([[-0.3057, -1.5471, -1.8411,  0.3489,  0.4888],
#         [-0.3927, -0.5330, -0.0332,  0.5013,  1.9109],
#         [-0.1904,  0.3057, -0.2462, -0.0511,  1.5072],
#         [ 0.0670,  1.9435, -0.2445,  0.9323,  1.4720]],
#        grad_fn=<EmbeddingBackward0>)
print(embedding.weight[0:5])
# tensor([[ 0.7542,  0.9717,  0.2982,  1.1881, -0.1999],
#         [-0.3057, -1.5471, -1.8411,  0.3489,  0.4888],
#         [ 0.0670,  1.9435, -0.2445,  0.9323,  1.4720],
#         [-0.1904,  0.3057, -0.2462, -0.0511,  1.5072],
#         [-0.3927, -0.5330, -0.0332,  0.5013,  1.9109]],
#        grad_fn=<SliceBackward0>)



# Batch of sentences (as indices)
sentences = torch.tensor([[1, 2, 3, 4],
                          [5, 6, 7, 0],
                          [8, 9, 0, 0]])

# Embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

# Get embeddings for the batch
embedded_sentences = embedding(sentences)

print(embedded_sentences.shape)  # Output: torch.Size([3, 4, embedding_dim])
print(embedded_sentences)
# Output:
# tensor([[[-0.4648,  1.9989,  1.4722,  0.3428, -1.2115],
#          [ 0.8178, -1.2466,  0.8559,  2.0523,  0.3668],
#          [ 0.9854, -0.8668, -0.3316,  1.2818, -0.0928],
#          [ 0.3453, -0.4938, -0.7498,  0.5813,  0.0731]],

#         [[-0.6073, -0.0538, -0.5604, -0.1414, -0.1443],
#          [ 0.1321, -0.4640, -0.0965,  1.1742,  0.0642],
#          [-1.0192,  0.4646,  1.5962,  0.1318,  0.1202],
#          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

#         [[ 1.7404, -0.6303, -1.0254,  0.5754,  0.7849],
#          [ 0.5697,  0.8108,  0.9563,  1.6761,  0.1987],
#          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],
#        grad_fn=<EmbeddingBackward0>)


import torch
import torch.nn as nn

# Initialize an Embedding layer
vocab_size = 10
embedding_dim = 3
embedding = nn.Embedding(vocab_size, embedding_dim)

tmp = embedding.weight

# Print initial embeddings for tokens 0 and 1
print("Before training:")
print(embedding.weight[0:2])

# Simulate training process
optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
for _ in range(100):
    loss = torch.sum(embedding.weight**2)  # Example loss function
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Print embeddings after training
print("\nAfter training:")
print(embedding.weight[0:2])