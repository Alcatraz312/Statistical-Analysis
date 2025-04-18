import numpy as np

T = np.array([
    [0.1, 0.3, 0.4, 0.2],
    [0.2, 0.2, 0.3, 0.3],
    [0.3, 0.3, 0.1, 0.3],
    [0.25, 0.25, 0.25, 0.25]
])

T = T.T

A = T - np.eye(4)

A = np.vstack([A, np.ones(4)])
b = np.zeros(5)
b[-1] = 1

pi = np.linalg.lstsq(a= A, b = b)[0]
print(pi)

for i in range(len(pi)):
    print(f"The steady state probability for particle to be in E{i + 1} : {pi[i]}")