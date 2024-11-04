import numpy as np

class LiquidNeuron:
    def __init__(self, time_constant):
        self.time_constant = time_constant
        self.state = 0
        self.connections = []

    def update(self, input_signal, dt):
        self.state += (input_signal - self.state) * dt / self.time_constant
        return self.state

class LiquidNN:
    def __init__(self, num_neurons, connectivity):
        self.neurons = [LiquidNeuron(np.random.uniform(0.1, 1.0)) for _ in range(num_neurons)]
        self.connect_neurons(connectivity)

    def connect_neurons(self, connectivity):
        for neuron in self.neurons:
            num_connections = int(np.random.uniform(1, len(self.neurons) * connectivity))
            neuron.connections = np.random.choice(self.neurons, num_connections, replace=False)

    def run(self, input_signal, duration, dt):
        output = []
        for t in np.arange(0, duration, dt):
            network_state = [n.update(input_signal(t), dt) for n in self.neurons]
            output.append(np.mean(network_state))
        return output

# Example usage
def input_signal(t):
    return np.sin(t * 2 * np.pi)

liquid_nn = LiquidNN(num_neurons=100, connectivity=0.1)
output = liquid_nn.run(input_signal, duration=10, dt=0.1)

print(f"Output: {output}")