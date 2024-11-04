import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the maze environment
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

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training function
def train_dqn(env, model, target_model, num_episodes, batch_size):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    memory = []

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                current_q_values = model(states).gather(1, actions.unsqueeze(1))
                # Use target network for next state Q-values
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = criterion(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

# Main execution
env = MazeEnv()
input_size = 16
output_size = 4

model = DQN(input_size, output_size)
target_model = DQN(input_size, output_size)
target_model.load_state_dict(model.state_dict())

num_episodes = 1000
batch_size = 32

train_dqn(env, model, target_model, num_episodes, batch_size)

# Test the trained model
state = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        q_values = model(torch.FloatTensor(state))
        action = q_values.argmax().item()
    state, reward, done = env.step(action)
    total_reward += reward
    print(f"Position: {env.position}, Action: {action}, Reward: {reward}")

print(f"Final position: {env.position}, Total reward: {total_reward}")