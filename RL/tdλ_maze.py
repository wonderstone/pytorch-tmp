import numpy as np
import random

# Grid world setup
grid_size = 5
target_position = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]
actions = ['up', 'down', 'left', 'right']

# Initialize value table and eligibility trace table
V = np.zeros((grid_size, grid_size))
E = np.zeros((grid_size, grid_size))  # Eligibility traces

# Parameters
learning_rate = 0.1
discount_factor = 0.9
lambda_val = 0.8
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

# TD(λ) for Policy Evaluation
for episode in range(episodes):
    state = (0, 0)  # Starting position
    E.fill(0)  # Reset eligibility traces at the start of each episode
    
    for step in range(max_steps):
        action = random.choice(actions)  # Random policy
        next_state = get_next_position(state, action)
        reward = get_reward(next_state)
        
        # TD error
        td_error = reward + discount_factor * V[next_state] - V[state]
        
        # Update eligibility trace for the current state
        E[state] += 1
        
        # Update all states' values and traces
        for s in np.ndindex(V.shape):
            V[s] += learning_rate * td_error * E[s]
            E[s] *= discount_factor * lambda_val  # Decay the eligibility trace
        
        state = next_state  # Move to the next state
        
        if state == target_position:
            break

# Display learned values
print("Learned State Values with TD(λ):")
print(V)