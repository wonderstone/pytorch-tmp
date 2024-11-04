import numpy as np

class TD0:
    def __init__(self, n_states, alpha=0.1, gamma=0.99):
        self.n_states = n_states
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.V = np.zeros(n_states)  # Initialize value function

    def update(self, state, next_state, reward):
        td_target = reward + self.gamma * self.V[next_state]
        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error

# Example usage
n_states = 5
td = TD0(n_states)

# Simulate some transitions
for _ in range(1000):
    state = np.random.randint(0, n_states)
    next_state = np.random.randint(0, n_states)
    reward = np.random.rand()
    td.update(state, next_state, reward)

print("Learned state values:")
print(td.V)