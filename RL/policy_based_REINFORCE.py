import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the maze environment
class MazeEnv:
    def __init__(self):
        self.maze = np.array([
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ])
        self.start = (0, 0)
        self.goal = (3, 3)
        self.position = self.start

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_position = tuple(np.add(self.position, directions[action]))
        
        if (0 <= new_position[0] < 4 and 0 <= new_position[1] < 4 and 
            self.maze[new_position] == 0):
            self.position = new_position

        done = (self.position == self.goal)
        reward = 10 if done else -1
        return self.position, reward, done

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# REINFORCE algorithm
def reinforce(env, policy, optimizer, num_episodes, gamma):
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            state = torch.FloatTensor(state)
            action_probs = policy(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)
            
            state, reward, done = env.step(action.item())
            rewards.append(reward)
            
            if done:
                break
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        loss = torch.stack(loss).sum()
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {sum(rewards)}')

# Main execution
env = MazeEnv()
input_size = 2  # x and y coordinates
output_size = 4  # up, right, down, left

policy = PolicyNetwork(input_size, output_size)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

num_episodes = 10000
gamma = 0.99

reinforce(env, policy, optimizer, num_episodes, gamma)