import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Maze environment (same as before)
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

# Define Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize the Actor and Critic
env = MazeEnv()
state_dim = env._get_state().shape[0]
action_dim = 4  # Up, Right, Down, Left

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    log_probs = []
    rewards = []
    values = []
    
    # Generate a trajectory
    for t in range(1000):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Actor forward pass for action probabilities
        probs = actor(state_tensor)
        action = torch.multinomial(probs, 1).item()
        
        # Critic forward pass for state value
        value = critic(state_tensor)
        
        # Take action in environment
        next_state, reward, done = env.step(action)
        
        log_prob = torch.log(probs[action])
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        
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

    # Actor loss: policy gradient with G as return
    actor_loss = -torch.sum(torch.stack(log_probs) * returns)
    
    # Critic loss: MSE between estimated values and returns
    critic_loss = nn.MSELoss()(torch.cat(values), returns)

    # Update Actor and Critic
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}, Steps: {t+1}")

print("Simplified Actor-Critic Training completed.")