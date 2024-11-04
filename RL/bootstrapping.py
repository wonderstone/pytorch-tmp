import numpy as np
import matplotlib.pyplot as plt

# Original dataset
data = np.array([5, 7, 12, 3, 9, 8, 11, 6, 4, 10])

# Number of bootstrap samples
n_bootstrap = 100000

# Function to calculate mean and variance
def calculate_stats(sample):
    return np.mean(sample), np.var(sample)

# Bootstrap resampling
bootstrap_means = np.zeros(n_bootstrap)
bootstrap_variances = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    # Resample with replacement
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    
    # Calculate statistics
    bootstrap_means[i], bootstrap_variances[i] = calculate_stats(bootstrap_sample)

# Calculate confidence intervals (95%)
mean_ci = np.percentile(bootstrap_means, [2.5, 97.5])
var_ci = np.percentile(bootstrap_variances, [2.5, 97.5])

print(f"Original Mean: {np.mean(data):.4f}")
print(f"Bootstrap Mean: {np.mean(bootstrap_means):.4f}")
print(f"Mean 95% CI: ({mean_ci[0]:.4f}, {mean_ci[1]:.4f})")
print(f"\nOriginal Variance: {np.var(data):.4f}")
print(f"Bootstrap Variance: {np.mean(bootstrap_variances):.4f}")
print(f"Variance 95% CI: ({var_ci[0]:.4f}, {var_ci[1]:.4f})")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(bootstrap_means, bins=100, edgecolor='black')
ax1.set_title('Bootstrap Distribution of Mean')
ax1.set_xlabel('Mean')
ax1.set_ylabel('Frequency')

ax2.hist(bootstrap_variances, bins=100, edgecolor='black')
ax2.set_title('Bootstrap Distribution of Variance')
ax2.set_xlabel('Variance')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()