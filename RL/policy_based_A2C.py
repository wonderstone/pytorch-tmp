import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

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

def worker(env, actor, critic, actor_optimizer, critic_optimizer, gamma, batch_size):
    log_probs = []
    values = []
    rewards = []
    state = env.reset()

    for _ in range(batch_size):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = actor(state_tensor)
        action = torch.multinomial(probs, 1).item()
        value = critic(state_tensor)
        
        next_state, reward, done = env.step(action)
        log_probs.append(torch.log(probs[action]))
        values.append(value)
        rewards.append(reward)

        state = next_state if not done else env.reset()

    # Calculate returns
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # Compute advantage
    advantages = returns - torch.cat(values)

    # Actor loss
    actor_loss = -torch.sum(torch.stack(log_probs) * advantages.detach())

    # Critic loss
    critic_loss = nn.MSELoss()(torch.cat(values), returns)

    # Update actor and critic
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

# A2C training loop with parallel environments
def train_a2c(num_workers=4, num_episodes=1000, batch_size=10, gamma=0.99):
    envs = [MazeEnv() for _ in range(num_workers)]
    actor = Actor(state_dim=16, action_dim=4)
    critic = Critic(state_dim=16)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(envs[i], actor, critic, actor_optimizer, critic_optimizer, gamma, batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("A2C Training completed.")

# Start A2C training
if __name__ == "__main__":
    mp.set_start_method('spawn')
    train_a2c()