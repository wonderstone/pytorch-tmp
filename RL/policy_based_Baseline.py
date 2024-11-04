import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Maze environment
class MazeEnv:
    def __init__(self):
        self.maze = np.array([
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 0]
        ])
        self.goal = (3, 3)
        self.reset()

    def reset(self):
        self.position = (0, 0)
        return self._get_state()

    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        move = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_position = (
            max(0, min(3, self.position[0] + move[0])),
            max(0, min(3, self.position[1] + move[1]))
        )
        if self.maze[new_position[1]][new_position[0]] == 0:
            self.position = new_position

        done = (self.position == self.goal)
        reward = 10 if done else -1
        return self._get_state(), reward, done

    def _get_state(self):
        state = np.zeros(16)
        state[self.position[1] * 4 + self.position[0]] = 1
        return state

# Define the Baseline Policy network
class BaselinePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BaselinePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.value_layer = nn.Linear(128, 1)  # Baseline value estimate

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        baseline_value = self.value_layer(x)
        return action_probs, baseline_value

# Training loop with only baseline
env = MazeEnv()
state_dim = env._get_state().shape[0]
action_dim = 4  # Up, Right, Down, Left

policy = BaselinePolicy(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.001)
num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    log_probs = []
    values = []
    rewards = []
    
    # Generate a trajectory
    for t in range(1000):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Get action probabilities and baseline value
        probs, baseline_value = policy(state_tensor)
        
        # Sample an action
        action = torch.multinomial(probs, 1).item()
        
        # Interact with the environment
        next_state, reward, done = env.step(action)
        
        # Store log probability, baseline value, and reward
        log_prob = torch.log(probs[action])
        log_probs.append(log_prob)
        values.append(baseline_value)
        rewards.append(reward)
        
        state = next_state
        
        if done:
            break

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    
    # Compute advantages (Advantage = Return - Baseline Value)
    values = torch.cat(values)
    advantages = returns - values.detach()
    
    # Policy Loss and Baseline Loss
    policy_loss = -(torch.stack(log_probs) * advantages).mean()
    baseline_loss = nn.MSELoss()(values, returns)

    # Total loss (combining both policy and baseline updates)
    total_loss = policy_loss + baseline_loss
    
    # Update policy network
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Print progress
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}, Steps: {t+1}")

print("Training finished!")

# use the trained policy to navigate the maze
state = env.reset()
done = False
print(env.maze)
# Keep navigating until the goal is reached
while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32)
    probs, _ = policy(state_tensor)
    action = torch.argmax(probs).item()
    state, _, done = env.step(action)
    
    print(np.array(state).reshape(4, 4))
    print("===>")
    print("")


