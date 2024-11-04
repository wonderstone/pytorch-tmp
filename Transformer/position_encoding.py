import numpy as np

def get_position_encoding(max_length, d_model):
    position = np.arange(max_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((max_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding

# Example usage
max_length = 5
d_model = 8
position_encoding = get_position_encoding(max_length, d_model)

print("Shape of position encoding:", position_encoding.shape)
print("\nPosition encoding matrix:")
print(np.round(position_encoding, decimals=3))