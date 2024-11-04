import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
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
        self.observation_space = np.zeros(16)
        self.action_space = np.arange(4)
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

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)  # Output a single value representing V(s)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return Categorical(logits=self.actor(x)), self.critic(x)

# Custom MazeEnv would be initialized similarly here

def worker(env, global_model, optimizer, gamma=0.99, T=1000):
    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.size)
    local_model.load_state_dict(global_model.state_dict())  # Initialize with global params

    state = env.reset()
    log_probs, values, rewards = [], [], []

    for t in range(T):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        dist, value = local_model(state_tensor)
        action = dist.sample()
        
        next_state, reward, done = env.step(action.item())

        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)

        state = next_state if not done else env.reset()

        if done or t == T - 1:
            # Compute returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)
            advantage = returns - values.squeeze()

            # Loss computation
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param.grad = local_param.grad  # Share gradients with global model
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())  # Sync with global

            log_probs, values, rewards = [], [], []

def train_a3c():
    env = MazeEnv()  # Or other environment
    global_model = ActorCritic(env.observation_space.shape[0], env.action_space.size)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-3)
    num_workers = 4  # Number of parallel workers

    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(env, global_model, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# start A3C training
if __name__ == "__main__":
    train_a3c()
    print("A3C Training completed.")