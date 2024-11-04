import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Custom Maze Environment
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

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return Categorical(logits=self.action_head(x))

# TRPO Helper Functions
def kl_divergence(policy_old, policy_new, states):
    dist_old = policy_old(states)
    dist_new = policy_new(states)
    kl = torch.distributions.kl.kl_divergence(dist_old, dist_new)
    return kl.mean()

def surrogate_loss(policy, states, actions, advantages, old_log_probs):
    dist = policy(states)
    new_log_probs = dist.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    return (ratio * advantages).mean()

def conjugate_gradient(policy, states, kl_gradient, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(kl_gradient)
    r = kl_gradient.clone()
    p = r.clone()
    rr_old = torch.dot(r, r)
    for _ in range(max_iter):
        Ap = fisher_vector_product(policy, states, p)
        alpha = rr_old / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rr_new = torch.dot(r, r)
        if rr_new < residual_tol:
            break
        p = r + (rr_new / rr_old) * p
        rr_old = rr_new
    return x

def fisher_vector_product(policy, states, vector, damping=0.1):
    vector = vector.detach()
    kl_div = kl_divergence(policy, policy, states)
    kl_grad = torch.autograd.grad(kl_div, policy.parameters(), create_graph=True)
    flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])
    kl_grad_vector_product = torch.dot(flat_kl_grad, vector)
    fisher_product = torch.autograd.grad(kl_grad_vector_product, policy.parameters())
    fisher_product_flat = torch.cat([g.contiguous().view(-1) for g in fisher_product])
    return fisher_product_flat + damping * vector

def trpo_update(policy, states, actions, advantages, old_log_probs, max_kl=0.01):
    # Get the surrogate loss and its gradient
    loss = surrogate_loss(policy, states, actions, advantages, old_log_probs)
    loss_grad = torch.autograd.grad(loss, policy.parameters())
    loss_grad_flat = torch.cat([g.view(-1) for g in loss_grad]).detach()
    
    # Compute the natural gradient direction using conjugate gradient
    step_direction = conjugate_gradient(policy, states, loss_grad_flat)
    
    # Scale the step size to satisfy the KL constraint
    step_size = torch.sqrt(2 * max_kl / (torch.dot(step_direction, fisher_vector_product(policy, states, step_direction)) + 1e-8))
    step_direction *= step_size

    # Apply the policy update
    params = list(policy.parameters())
    index = 0
    for param in params:
        num_params = param.numel()
        step = step_direction[index:index + num_params].view(param.size())
        param.data.add_(step)
        index += num_params

# Training Loop
def train_trpo(env, policy, num_episodes=100):
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        log_probs, rewards, states, actions = [], [], [], []
        done = False
        while not done:
            dist = policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state

        # Compute cumulative rewards and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - returns.mean()

        # Convert lists to tensors
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs = torch.stack(log_probs)

        # TRPO update
        trpo_update(policy, states, actions, advantages, log_probs)

        print(f"Episode {episode + 1}: Total Reward: {sum(rewards)}")

# Initialize environment and policy network
env = MazeEnv()
policy = PolicyNetwork(input_dim=16, action_dim=4)

# Train TRPO
train_trpo(env, policy)