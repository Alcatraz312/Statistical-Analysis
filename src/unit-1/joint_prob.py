# Importing required libraries
import numpy as np                      # for numerical operations and random sampling
import matplotlib.pyplot as plt         # for plotting
import seaborn as sns                   # for better-looking plots
from scipy.stats import norm, gaussian_kde  # for probability distribution functions and kernel density estimation

# Define possible values for two discrete random variables
a = [1, 2, 3]        # Possible values for variable x
b = [10, 20]         # Possible values for variable y

# Generate 1000 samples of x and y independently from specified probability distributions
x = np.random.choice([1, 2, 3], size=1000, p=[0.2, 0.5, 0.3])   # x sampled with given probabilities
y = np.random.choice([10, 20], size=1000, p=[0.6, 0.4])         # y sampled with given probabilities

# Define a function to count how often each value in content_list appears in array x
def Counter(x, content_list):
    freq = []  # to store frequency counts
    for i in content_list:
        s = 0
        for j in range(len(x)):
            if x[j] == i:
                s += 1
        freq.append(s)
    return dict(zip(content_list, freq))  # return a dictionary mapping values to their frequencies

# Total number of samples
N = len(x)

# Get frequency count of each value in x and y using the Counter function
x_count = Counter(x, a)
y_count = Counter(y, b)

# Calculate marginal probabilities for x and y by dividing frequency by total samples
P_x = {val: count / N for val, count in x_count.items()}
P_y = {val: count / N for val, count in y_count.items()}

# Compute joint probability distribution assuming independence:
# P(x, y) = P(x) * P(y)
joint_prob = {}
for x_val, p_x in P_x.items():
    for y_val, p_y in P_y.items():
        joint_prob[(x_val, y_val)] = p_x * p_y  # independent variables ⇒ product of marginals

# Show the final joint probability table
print(f"Joint probability for discrete case : {joint_prob}")


# Generate continuous independent data
x_cont = np.random.normal(loc=0, scale=1, size=1000)   # Standard normal
y_cont = np.random.normal(loc=5, scale=2, size=1000)   # Normal with mean=5, std=2

# Estimate marginal probability densities using Gaussian KDE
kde_x = gaussian_kde(x_cont)
kde_y = gaussian_kde(y_cont)

# Create a grid of (x, y) points
x_vals = np.linspace(min(x_cont), max(x_cont), 100)
y_vals = np.linspace(min(y_cont), max(y_cont), 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute joint density as product of marginals (since independent)
Z = np.outer(kde_x(x_vals), kde_y(y_vals))

print(f"Joint Probability for continous case : {Z}")

# Plot joint density as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(Z, cmap="viridis", xticklabels=10, yticklabels=10)
plt.title("Joint Probability Density (Continuous Case)")
plt.xlabel("y values")
plt.ylabel("x values")
plt.tight_layout()
plt.savefig("./images/figurejoint")