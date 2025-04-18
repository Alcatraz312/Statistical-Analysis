import numpy as np  # Import NumPy library for numerical operations

# Define the transition probability matrix T (T[i][j] is the probability of transitioning from Ei to Ej)
T = np.array([
    [0.1, 0.3, 0.4, 0.2],
    [0.2, 0.2, 0.3, 0.3],
    [0.3, 0.3, 0.1, 0.3],
    [0.25, 0.25, 0.25, 0.25]
])

# Transpose the matrix so we can write the equation in the form (Tᵀ - I)πᵀ = 0
T = T.T

# Subtract identity matrix to set up the homogeneous system (Tᵀ - I)πᵀ = 0
A = T - np.eye(4)  # np.eye(4) is a 4x4 identity matrix

# Add the normalization condition: π1 + π2 + π3 + π4 = 1
A = np.vstack([A, np.ones(4)])  # Stack a row of ones to add the normalization equation
b = np.zeros(5)                 # Initialize right-hand side vector b with 5 zeros
b[-1] = 1                       # Set the last value of b to 1 (normalization condition)

# Solve the overdetermined linear system Aπ = b using least squares
pi = np.linalg.lstsq(a=A, b=b, rcond=None)[0]  # [0] extracts the solution vector π

# Print the steady-state probability distribution
print(pi)

# Display the long-term probabilities for each energy state E1 through E4
for i in range(len(pi)):
    print(f"The steady state probability for particle to be in E{i + 1} : {pi[i]}")
