import numpy as np
from scipy.stats import binomtest

n = 100    # number of tosses of the coin
q = 0.6    # probability of landing heads on each toss

tosses = np.random.binomial(1, size = n, p = q)
num_heads = sum(tosses)

print(f"Number of heads = {num_heads} out of {n} tosses.")

test_result = binomtest(k = num_heads, n = n, p = 0.5, alternative= "two-sided")

print(f"P-value: {test_result.pvalue:.4f}")

if test_result.pvalue < 0.05:
    print("Reject the null hypothesis (coin is likely biased).")
else:
    print("Failed to reject the null hypothesis (Coin appears fair).")