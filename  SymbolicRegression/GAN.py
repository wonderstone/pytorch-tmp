import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp
import numpy as np
import random

# Define operator and operand sets
operators = ['+', '-', '*', '/']
operands = ['x', '1', '2', '3', '4', '5']

# Binary Tree Node class to represent the structure of an expression
class TreeNode:
    def __init__(self, value):
        self.value = value  # Operator or operand
        self.left = None    # Left child (TreeNode)
        self.right = None   # Right child (TreeNode)

# Function to generate a tree based on latent vector, with wrap-around if idx exceeds size
def generate_expression_tree_from_latent(latent_vector, max_depth):
    idx = 0
    latent_size = len(latent_vector)  # Size of the latent vector

    def build_tree(depth):
        nonlocal idx
        if depth == 0 or random.random() < 0.5:
            # Use latent vector to select an operand, with wrap-around
            value = operands[int(latent_vector[idx % latent_size] % len(operands))]
            idx += 1
            return TreeNode(value)
        else:
            # Use latent vector to select an operator, with wrap-around
            value = operators[int(latent_vector[idx % latent_size] % len(operators))]
            idx += 1
            node = TreeNode(value)
            node.left = build_tree(depth - 1)
            node.right = build_tree(depth - 1)
            return node
    
    return build_tree(max_depth)


# Function to convert a tree into a SymPy expression
def tree_to_sympy_expr(node):
    if node is None:
        return None
    if node.value in operators:
        left_expr = tree_to_sympy_expr(node.left)
        right_expr = tree_to_sympy_expr(node.right)
        return sp.sympify(f"({left_expr} {node.value} {right_expr})")
    else:
        return sp.sympify(node.value)

# Generator Model: Outputs latent vector that defines the tree structure
class Generator(nn.Module):
    def __init__(self, z_dim, latent_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim, latent_dim)  # z_dim -> latent_dim

    def forward(self, z):
        latent_vector = torch.relu(self.fc(z))
        return latent_vector

# Discriminator Model: Evaluates how well the generated expression fits the target
class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        out = torch.relu(self.fc1(error))
        out = torch.sigmoid(self.fc2(out))
        return out

# Example dataset: y = 2 * x^2 + 3 * x + 1
x_vals = np.linspace(-1, 1, 100)
y_vals = 2 * x_vals*2 + 3 * x_vals + 1

# Convert y_vals to torch tensor for the discriminator
y_true = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)

# Hyperparameters
latent_dim = 20  # Latent vector size
z_dim = 10  # Noise dimension for the generator
max_depth = 5  # Max depth of the expression trees
hidden_dim = 64  # Hidden size for both generator and discriminator

# Initialize generator and discriminator
generator = Generator(z_dim, latent_dim)
discriminator = Discriminator(hidden_dim)

optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

best_loss = float('inf')
best_expr = None

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    # Generate random noise and pass it through the generator
    z = torch.randn(1, z_dim)
    latent_vector = generator(z).detach().numpy().flatten()

    # Generate expression tree based on latent vector
    expr_tree = generate_expression_tree_from_latent(latent_vector, max_depth)
    expr = tree_to_sympy_expr(expr_tree)

    try:
        # Evaluate the expression
        y_pred = sp.lambdify('x', expr, 'numpy')(x_vals)

        # Check if y_pred contains invalid values like 'zoo', 'ComplexInfinity', 'nan', or 'inf'
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            raise ValueError(f"Invalid expression result (contains inf/nan): {expr}")

        # Fix: If y_pred is a scalar (constant), broadcast it to match x_vals
        if np.isscalar(y_pred):
            y_pred = np.full_like(x_vals, y_pred)

        # Convert y_pred to a PyTorch tensor and ensure it is 2D
        y_pred = torch.tensor(y_pred, dtype=torch.float32).unsqueeze(1)
        
        # Train Discriminator
        batch_size = y_pred.size(0)
        real_labels = torch.ones(batch_size, 1)  # Change size to match D_real
        fake_labels = torch.zeros(batch_size, 1)  # Change size to match D_fake

        D_real = discriminator(y_true, y_true)
        D_fake = discriminator(y_true, y_pred)

        D_loss_real = loss_fn(D_real, real_labels)
        D_loss_fake = loss_fn(D_fake, fake_labels)
        D_loss = D_loss_real + D_loss_fake

        optimizer_d.zero_grad()
        D_loss.backward()
        optimizer_d.step()

        # Train Generator
        D_fake = discriminator(y_true, y_pred)
        G_loss = loss_fn(D_fake, real_labels)

        optimizer_g.zero_grad()
        G_loss.backward()
        optimizer_g.step()

        # Track best-fitting expression
        total_loss = D_loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            best_expr = expr

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{num_epochs} - D_loss: {D_loss.item()} - G_loss: {G_loss.item()}')
            print(f'Generated Expression: {expr}')
    
    except ValueError as e:
        print(f'Epoch {epoch}/{num_epochs} - Error: {e}')
    except Exception as e:
        print(f"Error: {e}")

# Final best expression
print(f'Best Expression: {best_expr} with loss: {best_loss}')