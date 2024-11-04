import numpy as np
import random

# Environment setup
grid_size = 5
target_position = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]
actions = ['up', 'down', 'left', 'right']

# Value table initialization
V = np.zeros((grid_size, grid_size))

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 1000
max_steps = 100

def get_next_position(position, action):
    if action == 'up':
        return (max(position[0] - 1, 0), position[1])
    elif action == 'down':
        return (min(position[0] + 1, grid_size - 1), position[1])
    elif action == 'left':
        return (position[0], max(position[1] - 1, 0))
    elif action == 'right':
        return (position[0], min(position[1] + 1, grid_size - 1))
    return position

def get_reward(position):
    if position == target_position:
        return 10
    elif position in obstacles:
        return -10
    else:
        return -1

# TD(0) for Policy Evaluation
for episode in range(episodes):
    state = (0, 0)  # Starting position
    for step in range(max_steps):
        # Random policy: select an action at random
        action = random.choice(actions)
        
        # Get next state and reward
        next_state = get_next_position(state, action)
        reward = get_reward(next_state)
        
        # TD(0) update rule
        V[state[0], state[1]] = V[state[0], state[1]] + \
            learning_rate * (reward + discount_factor * V[next_state[0], next_state[1]] - V[state[0], state[1]])
        
        # Move to next state
        state = next_state
        
        # Break if reached target
        if state == target_position:
            break

# Display the learned value table
print("Learned State Values:")
print(V)