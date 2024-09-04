import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, env
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, replay_buffer_size=1000, batch_size=64):
        self.env = env
        self.state_dim = env.size[0] * env.size[1]
        self.action_dim = 4
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.model = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        state_tensor = torch.zeros(self.state_dim)
        index = state[0] * self.env.size[1] + state[1]
        state_tensor[index] = 1.0
        return state_tensor

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return np.random.choice(4)
        else:
            state_tensor = self.state_to_tensor(state)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.stack([self.state_to_tensor(s) for s, _, _, _, _ in batch])
        action_batch = torch.tensor([a for _, a, _, _, _ in batch])
        reward_batch = torch.tensor([r for _, _, r, _, _ in batch])
        next_state_batch = torch.stack([self.state_to_tensor(ns) for _, _, _, ns, _ in batch])
        done_batch = torch.tensor([d for _, _, _, _, d in batch])

        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.discount_factor * next_q_values * torch.logical_not(done_batch)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay

if __name__ == '__main__':
    # 初始化迷宫环境和 DQN 代理
    env = env.MazeEnv()
    agent = DQNAgent(env)

    # 训练 DQN 代理
    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update_model()
            state = next_state
        
        agent.decay_exploration()

    # 测试 DQN 代理
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        env.render()