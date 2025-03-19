# central_limit_theorem.py
import numpy as np
import matplotlib.pyplot as plt
from coin_flip import t  # Import the coin_flip array

# Close any open figures
plt.close('all')

# Calculate the distribution
n = 1000
# (Alternatively, you could generate random numbers: t = np.random.rand(n))
trials = np.arange(1, n+1)
heads = 0
P_heads = np.ones_like(t, dtype=float)  # initialize array with the same shape as t

for ii in range(n):
    if t[ii] < 0.5:
        heads += 1
    P_heads[ii] = heads / (ii + 1)  # note: Python indexing starts at 0

# Animate plot of trial number vs. probability of heads
plt.figure(1)
for ii in range(n):
    plt.clf()  # Clear the current figure
    plt.plot(trials[:ii+1], P_heads[:ii+1], linewidth=2)
    plt.xlim([-50, 1000])
    plt.ylim([0, 1])
    plt.xlabel('Trial Number')
    plt.ylabel('Probability of Heads after n trials')
    plt.title('Trial Number vs Percent Heads')
    plt.pause(0.001)  # Pause briefly to update the plot

# Save the final figure
plt.savefig("../figures/law-of-large-numbers.pdf")
plt.show()
