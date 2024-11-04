import numpy as np
import random
from deap import base, creator, tools, algorithms

# Define the target function
def target_function(x):
    return 2 * x**2 + 3 * x + 1

# Define a simple expression evaluator
def evaluate_expression(expr, x):
    # Evaluate the expression using Python's eval function
    return eval(expr)

# Fitness function for evaluating expressions
def evaluate_individual(individual):
    # Convert individual (list of tokens) to a string expression
    expression = ''.join(individual)
    
    # Generate input values for evaluation
    x_values = np.linspace(-10, 10, 100)
    
    # Calculate target values
    target_values = target_function(x_values)
    
    # Calculate predicted values using the generated expression
    predicted_values = np.array([evaluate_expression(expression.replace('x', str(x)), x) for x in x_values])
    
    # Calculate Mean Squared Error
    mse = np.mean((predicted_values - target_values) ** 2)
    
    return (mse,)

# Genetic Programming setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", lambda: [random.choice(['2*x**2', '3*x', '1', '+', '-', '*', '(', ')']) for _ in range(10)])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register evaluation and genetic operators
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Generate initial population and run genetic algorithm
population_size = 20
generations = 50

population = toolbox.population(n=population_size)

for gen in range(generations):
    # Evaluate individuals
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select next generation individuals
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:  # Crossover probability
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:  # Mutation probability
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Replace old population by offspring
    population[:] = offspring

# Best individual after evolution
best_individual = tools.selBest(population, 1)[0]
best_expression = ''.join(best_individual)
print("Best Expression:", best_expression)

# Evaluate best expression against the target function
x_test = np.linspace(-10, 10, 100)
predicted_values = np.array([evaluate_expression(best_expression.replace('x', str(x)), x) for x in x_test])
target_values = target_function(x_test)

# Plotting results (optional)
import matplotlib.pyplot as plt

plt.plot(x_test, target_values, label='Target Function: $y=2x^2 + 3x + 1$', color='blue')
plt.plot(x_test, predicted_values, label='Predicted Function', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Target vs Predicted Function')
plt.grid()
plt.show()