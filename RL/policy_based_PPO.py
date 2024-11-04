import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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

# Actor-Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

# PPO training function
def train_ppo(env, actor, critic, actor_optimizer, critic_optimizer, gamma=0.99, epsilon=0.2, K_epochs=4, T=1000):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    log_probs = []
    dones = []

    # Collect trajectories
    for t in range(T):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Sample action from the policy
        probs = actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Take the action in the environment
        next_state, reward, done = env.step(action.item())

        # Store experience
        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        dones.append(done)

        # Move to the next state
        state = next_state if not done else env.reset()

    # Convert lists to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions)
    log_probs = torch.stack(log_probs)
    
    # Calculate discounted rewards and advantages
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    with torch.no_grad():
        values = critic(states).squeeze()
    advantages = returns - values

    # Update policy K epochs
    for _ in range(K_epochs):
        # Recalculate probabilities and values for the current policy
        new_probs = actor(states)
        dist = Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        
        # Calculate the probability ratio
        ratio = torch.exp(new_log_probs - log_probs.detach())

        # PPO clip objective
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        # Update Actor Network
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update Critic Network with fresh values
        values = critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    print("PPO Training Step completed.")

# Instantiate environment and models
env = MazeEnv()
actor = Actor(state_dim=16, action_dim=4)
critic = Critic(state_dim=16)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# Train PPO
train_ppo(env, actor, critic, actor_optimizer, critic_optimizer)